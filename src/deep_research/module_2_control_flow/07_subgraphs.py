"""
============================================================================
Stage 9: Subgraphs - encapsulating an agent as a reusable graph
============================================================================

Why subgraphs exist
-------------------
By Stage 8 we already have ~5 nodes in our graph and the file is
getting cluttered. The capstone has 7 agent types and 12-28 invocations
per query. If we wrote it all in one big graph file, it'd be
unreadable and untestable.

A SUBGRAPH lets you take a self-contained mini-graph (e.g. "the
Searcher agent: search -> filter -> summarize") and use it as a
SINGLE NODE inside a larger parent graph. The parent doesn't see
the inner nodes; it just sees one box.

Benefits:
  * Encapsulation - the searcher's internals can change without
    touching the parent graph
  * Reusability - same subgraph used in multiple parent graphs
  * Testability - test each agent's subgraph in isolation
  * Clarity - parent graph becomes 5 nodes instead of 50

This is exactly how the capstone is structured: Planner, Searcher,
Critic, Fact-Checker, Synthesizer are each their own subgraph
compiled separately, and the top-level "swarm" graph wires them.

Two ways to use a subgraph
--------------------------
Way 1: SAME state schema as parent
    If the subgraph's state schema overlaps with the parent's, you
    can use the compiled subgraph directly as a node:

        parent_builder.add_node("searcher_agent", compiled_searcher_subgraph)

    LangGraph passes the parent's state in, the subgraph runs, and
    its state updates are merged back into the parent. Easiest case.

Way 2: DIFFERENT state schema (input/output transformation)
    More common in real systems. The subgraph has its own focused
    state (e.g. {"sub_question": str, "summary": str}) different
    from the parent's. Then you wrap the call:

        async def searcher_node(state: ParentState) -> dict:
            inner_input = {"sub_question": state["pending_questions"][0]}
            inner_output = await searcher_agent.ainvoke(inner_input)
            return {"findings": [inner_output["summary"]]}

    This is "input/output transformation." It keeps each subgraph's
    state minimal (good for context isolation - same idea as Send
    payloads in Stage 6).

We'll demonstrate Way 2 since it's the more useful pattern.

Outer graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> P[planner]
        P -.Send fan-out.-> A1[searcher_agent #1<br/>= subgraph]
        P -.-> A2[searcher_agent #2<br/>= subgraph]
        P -.-> A3[searcher_agent #N<br/>= subgraph]
        A1 --> SY[synthesize]
        A2 --> SY
        A3 --> SY
        SY --> E([END])
    ```

Inner subgraph (one searcher_agent expanded):

    ```mermaid
    flowchart LR
        IS([START]) --> SE[search<br/>Tavily]
        SE --> FI[filter<br/>drop low-quality]
        FI --> CO[compress<br/>LLM summary]
        CO --> IE([END])
    ```

Two ways to use a subgraph (concept):

    ```mermaid
    flowchart TB
        subgraph Way1["Way 1: same state schema<br/>drop in directly"]
            P1[parent graph] -->|state passes through| C1["compiled_subgraph<br/>(used as node)"]
            C1 --> P1B[next parent node]
        end
        subgraph Way2["Way 2: different state schema<br/>wrapper translates I/O (RECOMMENDED)"]
            P2[parent state] --> WR1["wrapper node<br/>translate IN"]
            WR1 -->|"agent.invoke(small dict)"| C2[subgraph]
            C2 --> WR2["wrapper node<br/>translate OUT"]
            WR2 --> P2B[parent state update]
        end
    ```
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


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tavily = TavilySearch(max_results=4, search_depth="basic")


# ===========================================================================
# PART A: THE INNER SUBGRAPH (the "searcher agent")
# ===========================================================================
# A self-contained 3-node mini-graph that:
#   1) does a Tavily search
#   2) filters out low-quality results
#   3) compresses what's left into a single summary string
#
# Its state is FOCUSED - just the inputs and outputs of THIS agent.
# It does NOT know about `query`, `final_summary`, or anything else
# in the parent graph. That's the point.

# ---------------------------------------------------------------------------
# A.1 Inner state - small, focused, no reducers needed (single writer per field)
# ---------------------------------------------------------------------------
class SearcherState(TypedDict):
    sub_question: str         # input
    raw_results: list[dict]   # written by step 1, read by step 2
    kept_results: list[dict]  # written by step 2, read by step 3
    summary: str              # written by step 3 (the final output)


# ---------------------------------------------------------------------------
# A.2 Inner nodes
# ---------------------------------------------------------------------------
def search_step(state: SearcherState) -> dict:
    print(f"  [searcher.search] {state['sub_question'][:60]!r}")
    raw = tavily.invoke({"query": state["sub_question"]})
    return {"raw_results": raw.get("results", [])}

def filter_step(state: SearcherState) -> dict:
    """Drop results with empty content or very short snippets.
    In production this is where domain-authority weighting goes."""
    kept = [
        r for r in state["raw_results"]
        if r.get("content") and len(r["content"]) > 80
    ]
    print(f"  [searcher.filter] kept {len(kept)}/{len(state['raw_results'])}")
    return {"kept_results": kept}

def compress_step(state: SearcherState) -> dict:
    """Use the LLM to compress N snippets into one tight paragraph."""
    snippets = "\n\n".join(
        f"- ({r.get('title','?')}) {r.get('content','')[:300]}"
        for r in state["kept_results"]
    )
    response = llm.invoke(
        "Compress the snippets below into ONE focused paragraph "
        "answering the sub-question. Cite source titles in brackets.\n\n"
        f"Sub-question: {state['sub_question']}\n\nSnippets:\n{snippets}"
    )
    print(f"  [searcher.compress] -> {len(response.content)} chars")
    return {"summary": response.content}


# ---------------------------------------------------------------------------
# A.3 Build & compile the subgraph
# ---------------------------------------------------------------------------
# This compiles into a runnable Pregel object - the same kind of object
# our top-level graphs have been. You can call .invoke() on it directly
# for unit-testing the searcher in isolation.
searcher_builder = StateGraph(SearcherState)
searcher_builder.add_node("search",   search_step)
searcher_builder.add_node("filter",   filter_step)
searcher_builder.add_node("compress", compress_step)
searcher_builder.add_edge(START, "search")
searcher_builder.add_edge("search", "filter")
searcher_builder.add_edge("filter", "compress")
searcher_builder.add_edge("compress", END)

searcher_agent = searcher_builder.compile()
# `searcher_agent` is now a reusable, testable, drop-in agent.


# ===========================================================================
# PART B: THE OUTER GRAPH (orchestrates N searcher agents)
# ===========================================================================

# ---------------------------------------------------------------------------
# B.1 Outer state
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    query: str
    sub_questions: Annotated[list[str], add]
    summaries: Annotated[list[str], add]   # one per searcher agent
    final_answer: str


# ---------------------------------------------------------------------------
# B.2 Planner (same idea as Stage 6)
# ---------------------------------------------------------------------------
class SubQuestions(BaseModel):
    questions: list[str] = Field(
        description="2-3 focused sub-questions for parallel research."
    )

planner_llm = llm.with_structured_output(SubQuestions)

def planner_node(state: ResearchState) -> dict:
    print(f"[outer.planner] {state['query']!r}")
    result: SubQuestions = planner_llm.invoke(
        f"Decompose into 2-3 sub-questions: {state['query']}"
    )
    return {"sub_questions": result.questions}


# ---------------------------------------------------------------------------
# B.3 The searcher AGENT NODE - this is the subgraph in action
# ---------------------------------------------------------------------------
# This node is in the OUTER graph. It receives a tiny payload from
# Send(...) (Stage 6 pattern), translates it into the SearcherState
# shape, calls the inner subgraph as if it were any function, and
# translates the inner output back into the outer state.
#
# This is "input/output transformation" - the cleanest way to use
# subgraphs when their state schema differs from the parent's.
class SearcherSpawnInput(TypedDict):
    """Payload passed to each spawned searcher agent via Send."""
    sub_question: str

def searcher_agent_node(state: SearcherSpawnInput) -> dict:
    # 1. Translate input: outer payload -> inner state shape
    inner_input: SearcherState = {
        "sub_question": state["sub_question"],
        "raw_results": [],
        "kept_results": [],
        "summary": "",
    }

    # 2. Run the subgraph. .invoke() works exactly like on top-level
    #    graphs - it returns the final state dict of the subgraph.
    inner_final = searcher_agent.invoke(inner_input)

    # 3. Translate output: inner state -> outer state update.
    #    Note we wrap in a list because `summaries` has the `add` reducer
    #    (concatenate). Returning [...] from N parallel agents stitches
    #    all their summaries together.
    return {"summaries": [inner_final["summary"]]}


# ---------------------------------------------------------------------------
# B.4 Dispatcher - fan out one searcher agent per sub-question (Stage 6)
# ---------------------------------------------------------------------------
def dispatch_searchers(state: ResearchState) -> list[Send]:
    print(f"[outer.dispatch] -> {len(state['sub_questions'])} parallel agents")
    return [
        Send("searcher_agent", {"sub_question": sq})
        for sq in state["sub_questions"]
    ]


# ---------------------------------------------------------------------------
# B.5 Final synthesizer
# ---------------------------------------------------------------------------
def synthesize_node(state: ResearchState) -> dict:
    summaries_block = "\n\n---\n\n".join(state["summaries"])
    response = llm.invoke(
        "You are a research synthesizer. Combine the per-sub-question "
        "summaries below into a single 5-7 sentence answer to the user's "
        "query. Preserve any source citations.\n\n"
        f"Query: {state['query']}\n\nSummaries:\n{summaries_block}"
    )
    return {"final_answer": response.content}


# ---------------------------------------------------------------------------
# B.6 Wire the outer graph
# ---------------------------------------------------------------------------
outer = StateGraph(ResearchState)
outer.add_node("planner",         planner_node)
outer.add_node("searcher_agent",  searcher_agent_node)  # <- subgraph wrapper
outer.add_node("synthesize",      synthesize_node)

outer.add_edge(START, "planner")
outer.add_conditional_edges("planner", dispatch_searchers, ["searcher_agent"])
outer.add_edge("searcher_agent", "synthesize")
outer.add_edge("synthesize", END)

graph = outer.compile()

graph.get_graph().draw_mermaid_png(output_file_path="src/deep_research/module_2_control_flow/07_subgraphs.png")

# ---------------------------------------------------------------------------
# 7. RUN
# ---------------------------------------------------------------------------
# Notice the print indentation in the output:
#   [outer.*]    = outer graph nodes
#   [searcher.*] = INNER subgraph nodes (indented 2 spaces by us, but
#                  the agent is also runnable on its own - try the
#                  block at the very bottom)
if __name__ == "__main__":
    initial: ResearchState = {
        "query": "What are the strongest open-source vector databases in 2025 and how do they compare?",
        "sub_questions": [],
        "summaries": [],
        "final_answer": "",
    }
    final = graph.invoke(initial)

    print("\n=== RESULT ===")
    print(f"sub-questions: {len(final['sub_questions'])}")
    print(f"summaries:     {len(final['summaries'])}")
    print(f"\n{final['final_answer']}\n")

    # ------------------------------------------------------------------
    # BONUS: a subgraph is ALSO runnable on its own (great for tests).
    # ------------------------------------------------------------------
    print("=== BONUS: running the searcher subgraph standalone ===")
    standalone = searcher_agent.invoke({
        "sub_question": "What is Qdrant and what makes it different from Pinecone?",
        "raw_results": [],
        "kept_results": [],
        "summary": "",
    })
    print(f"\nstandalone summary:\n{standalone['summary']}")
