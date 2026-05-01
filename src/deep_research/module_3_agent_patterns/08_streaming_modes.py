"""
============================================================================
Stage 10: Streaming - watching a graph run in real time
============================================================================

Until now we've only used `graph.invoke(state)` which BLOCKS until the
whole graph finishes and then returns the final state. That's fine for
a CLI but useless for a UI: with parallel searchers running for 30+
seconds, the user stares at a frozen screen.

LangGraph's `.stream()` method yields events AS THEY HAPPEN, so we
can:
  * Render an "agent activity feed" (which agent is running, what tool
    it just called, how many tokens so far)
  * Stream LLM tokens character-by-character to the UI
  * Emit our own custom events from inside nodes ("memory cache hit",
    "trust score: 87")

This stage demonstrates the FOUR streaming modes that LangGraph
exposes. They're not alternatives - you usually combine them. The
PROJECT2_PLAN.md activity stream uses all four.

The four streaming modes
------------------------
1. "values"   - after every node finishes, yield the FULL state.
                Easy to use; bandwidth-heavy on big states.
2. "updates"  - after every node finishes, yield only the DELTA
                (what THAT node returned). Cheaper than "values".
3. "messages" - yield individual LLM TOKENS as they arrive from
                the model. This is how you stream chat responses.
4. "custom"   - yield arbitrary events you emit from inside nodes
                using `get_stream_writer()`. Perfect for "tool_call",
                "cache_hit", "cost_update" notifications.

You select modes by passing `stream_mode=` to `.stream()`. You can
pass a single string ("updates") or a list (["updates", "messages",
"custom"]) - in the latter case each yielded item is a (mode, data)
tuple so you can route it appropriately.

Graph topology (mermaid) - same as Stage 6:

    ```mermaid
    flowchart LR
        S([START]) --> P[planner<br/>emits 'planning_started/done']
        P -.Send.-> S1[searcher #1<br/>emits 'tavily_started/finished']
        P -.-> S2[searcher #2]
        P -.-> S3[searcher #N]
        S1 --> SU[summarizer<br/>token streaming via 'messages']
        S2 --> SU
        S3 --> SU
        SU --> E([END])
    ```

The four streaming modes side-by-side:

    ```mermaid
    flowchart TB
        G[graph.stream] --> V["mode='values'<br/>full state after each step"]
        G --> U["mode='updates'<br/>only the delta per step"]
        G --> M["mode='messages'<br/>per-token AIMessageChunk"]
        G --> C["mode='custom'<br/>writer({}) events from nodes"]
        V --> UC["combined =<br/>list of (mode, data) tuples<br/>each item routed to UI"]
        U --> UC
        M --> UC
        C --> UC
    ```

Custom events flow (`get_stream_writer()`):

    ```mermaid
    flowchart LR
        N[searcher_node] -->|"writer({event: 'tavily_started'})"| SW[stream writer]
        SW -->|yielded by graph.stream| UI["consumer loop<br/>(UI / log / SSE)"]
    ```

We reuse the parallel-fanout pattern from Stage 6 because it shows off
ALL FOUR modes at once.
============================================================================
"""

import os
from typing import TypedDict, Annotated
from operator import add

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
# NEW: get_stream_writer is how nodes emit "custom" events.
from langgraph.config import get_stream_writer


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tavily = TavilySearch(max_results=2, search_depth="basic")


# ---------------------------------------------------------------------------
# 1. STATE (same as Stage 6)
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    query: str
    sub_questions: Annotated[list[str], add]
    findings: Annotated[list[str], add]
    final_summary: str


# ---------------------------------------------------------------------------
# 2. PLANNER
# ---------------------------------------------------------------------------
class SubQuestions(BaseModel):
    questions: list[str] = Field(
        description="2-3 focused sub-questions for parallel research."
    )

planner_llm = llm.with_structured_output(SubQuestions)

def planner_node(state: ResearchState) -> dict:
    # We can emit custom events from ANY node. get_stream_writer()
    # returns a function we call with arbitrary dicts; the consumer
    # of `.stream(stream_mode="custom")` will receive them in order.
    # If streaming is disabled, get_stream_writer() returns a no-op.
    writer = get_stream_writer()
    writer({"event": "planning_started", "query": state["query"]})

    result = planner_llm.invoke(
        f"Decompose into 2-3 focused sub-questions: {state['query']}"
    )

    writer({"event": "planning_done", "n_sub_questions": len(result.questions)})
    return {"sub_questions": result.questions}


# ---------------------------------------------------------------------------
# 3. DISPATCHER + SEARCHER (Stage 6 fan-out, with custom events)
# ---------------------------------------------------------------------------
def dispatch_searchers(state: ResearchState) -> list[Send]:
    return [Send("searcher", {"sub_question": sq}) for sq in state["sub_questions"]]

class SearcherInput(TypedDict):
    sub_question: str

def searcher_node(state: SearcherInput) -> dict:
    writer = get_stream_writer()
    sub_q = state["sub_question"]

    # Emit a "tool_started" event - this is exactly what the swarm UI
    # consumes to show "Searcher is calling Tavily for: ...".
    writer({"event": "tavily_started", "sub_question": sub_q})

    raw = tavily.invoke({"query": sub_q})
    results = raw.get("results", [])

    writer({
        "event": "tavily_finished",
        "sub_question": sub_q,
        "n_results": len(results),
    })

    findings = [
        f"Q: {sub_q}\n  -> {r.get('content', '')[:180]}"
        for r in results
    ]
    return {"findings": findings}


# ---------------------------------------------------------------------------
# 4. SUMMARIZER - the node we want to stream TOKENS from
# ---------------------------------------------------------------------------
# When we stream with mode="messages", LangGraph yields a stream of
# (chunk, metadata) pairs for EVERY LLM call inside any node. Each
# `chunk` is an AIMessageChunk (LangChain's incremental token type).
# The text is in chunk.content.
#
# Note: structured-output calls are NOT token-streamable (the model
# emits the whole JSON object atomically). Free-form .invoke() IS
# streamable. So this is the place we get streaming tokens.
def summarizer_node(state: ResearchState) -> dict:
    findings_block = "\n\n".join(state["findings"])
    response = llm.invoke(
        "Write a 4-sentence answer using ONLY these findings.\n\n"
        f"Query: {state['query']}\n\nFindings:\n{findings_block}"
    )
    return {"final_summary": response.content}


# ---------------------------------------------------------------------------
# 5. WIRE
# ---------------------------------------------------------------------------
builder = StateGraph(ResearchState)
builder.add_node("planner",    planner_node)
builder.add_node("searcher",   searcher_node)
builder.add_node("summarizer", summarizer_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", dispatch_searchers, ["searcher"])
builder.add_edge("searcher", "summarizer")
builder.add_edge("summarizer", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 6. RUN: demonstrate each mode separately, then all together
# ---------------------------------------------------------------------------
def run_updates_mode():
    """`updates`: yields {node_name: state_delta} after each node."""
    print("\n=== mode='updates' ===")
    print("(Yields what each node CHANGED, not the full state. Cheap & useful.)\n")

    initial = {"query": "What are the top open-source vector databases in 2025?",
               "sub_questions": [], "findings": [], "final_summary": ""}

    for chunk in graph.stream(initial, stream_mode="updates"):
        # `chunk` is a dict like {"planner": {"sub_questions": [...]}}
        for node_name, delta in chunk.items():
            keys = list(delta.keys()) if isinstance(delta, dict) else "?"
            print(f"  [update] {node_name} -> wrote {keys}")


def run_values_mode():
    """`values`: yields the FULL state after every node."""
    print("\n=== mode='values' ===")
    print("(Yields the whole state after each step. Useful for debugging or replays.)\n")

    initial = {"query": "What is LangGraph good for?",
               "sub_questions": [], "findings": [], "final_summary": ""}

    for state in graph.stream(initial, stream_mode="values"):
        # `state` is the full ResearchState dict at this checkpoint.
        n_sq = len(state.get("sub_questions", []))
        n_f  = len(state.get("findings", []))
        print(f"  [snapshot] sub_qs={n_sq}, findings={n_f}, "
              f"summary={'yes' if state.get('final_summary') else 'no'}")


def run_messages_mode():
    """`messages`: yields LLM tokens as they arrive."""
    print("\n=== mode='messages' ===")
    print("(Streams TOKENS from the summarizer's LLM call - watch it type.)\n")

    initial = {"query": "Explain reducers in LangGraph briefly.",
               "sub_questions": [], "findings": [], "final_summary": ""}

    print("  ", end="", flush=True)
    for chunk, metadata in graph.stream(initial, stream_mode="messages"):
        # Filter to only the summarizer's tokens (otherwise we'd see
        # token streams from the structured-output planner call too,
        # which arrive as a single big chunk anyway).
        if metadata.get("langgraph_node") == "summarizer":
            text = chunk.content
            if text:
                print(text, end="", flush=True)
    print()


def run_custom_mode():
    """`custom`: yields events emitted via get_stream_writer()."""
    print("\n=== mode='custom' ===")
    print("(Yields events that nodes emit explicitly. Perfect for UI activity feeds.)\n")

    initial = {"query": "Summarize the differences between Pinecone and Qdrant.",
               "sub_questions": [], "findings": [], "final_summary": ""}

    for event in graph.stream(initial, stream_mode="custom"):
        # `event` is exactly the dict we passed to writer(...)
        print(f"  [custom] {event}")


def run_combined_mode():
    """All modes at once. Each yielded item is (mode, data)."""
    print("\n=== mode=['updates', 'custom', 'messages'] (combined) ===")
    print("(Real-world UIs use this: structural updates + custom events + token stream.)\n")

    initial = {"query": "Compare LangGraph vs CrewAI in one paragraph.",
               "sub_questions": [], "findings": [], "final_summary": ""}

    for mode, data in graph.stream(
        initial,
        stream_mode=["updates", "custom", "messages"],
    ):
        if mode == "updates":
            print(f"\n  [update] {list(data.keys())}")
        elif mode == "custom":
            print(f"  [custom] {data}")
        elif mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "summarizer" and chunk.content:
                print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    run_updates_mode()
    run_values_mode()
    run_messages_mode()
    run_custom_mode()
    run_combined_mode()
