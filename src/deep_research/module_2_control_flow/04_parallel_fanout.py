"""
============================================================================
Stage 6: Parallel fan-out with the Send API
============================================================================

This is the FIRST file where we go beyond a linear pipeline. We're going
to take ONE planner output (a list of N sub-questions) and run N searcher
nodes IN PARALLEL, then collect all their findings.

The new primitive: `Send`
-------------------------
A Send is a "go run THIS node with THIS specific input" instruction.
Returning a LIST of Sends from a routing function tells LangGraph:
"fan out — run all of these in parallel, then continue."

Before Send, our edges said "after node A, go to node B." That's
1-to-1. Send lets us say "after node A, dispatch K copies of node B,
each with different input." That's 1-to-N.

Key mental shift
----------------
With Send, each spawned node receives its OWN private input dict (NOT
the shared state). This is "context isolation" - each searcher only
sees its own sub-question, not the entire research context. This is
literally why multi-agent systems beat single agents: less context
pollution per worker.

Why reducers FINALLY earn their keep
------------------------------------
Stage 2/3 introduced `Annotated[list[str], add]` but the reducer never
actually had to merge anything because we only had one writer. NOW we
have N parallel writers all returning {"findings": [...]}. Without
the reducer, the last one to finish wins and the others vanish.
With the reducer, every searcher's findings get appended.

Graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> P[planner]
        P -.dispatch_searchers<br/>list of Sends.-> S1[searcher #1]
        P -.-> S2[searcher #2]
        P -.-> S3[searcher #N]
        S1 --> SU[summarizer]
        S2 --> SU
        S3 --> SU
        SU --> E([END])
    ```

Anatomy of a Send (concept):

    ```mermaid
    flowchart LR
        D["dispatcher<br/>(routing fn)"] -->|"Send('searcher', {sub_question: 'q1'})"| W1[searcher #1]
        D -->|"Send('searcher', {sub_question: 'q2'})"| W2[searcher #2]
        D -->|"Send('searcher', {sub_question: 'q3'})"| W3[searcher #3]
        W1 -.context isolated.- W2
        W2 -.context isolated.- W3
    ```

Reducer merging under parallel writes:

    ```mermaid
    flowchart TB
        W1["searcher #1<br/>returns {findings: ['a']}"]
        W2["searcher #2<br/>returns {findings: ['b']}"]
        W3["searcher #3<br/>returns {findings: ['c']}"]
        R{{"add reducer<br/>concat in arrival order"}}
        S["state.findings = ['a','b','c']<br/>(order non-deterministic)"]
        W1 --> R
        W2 --> R
        W3 --> R
        R --> S
    ```

The "split" happens via a CONDITIONAL EDGE that returns a list of Sends.
The "join" is automatic - LangGraph waits for all parallel branches to
finish, merges their state updates with the reducers, then continues.
============================================================================
"""

import os
from typing import TypedDict, Annotated
from operator import add

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# NEW IMPORT: `Send` is the only new thing we need from LangGraph.
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ---------------------------------------------------------------------------
# 0. SETUP (same as Stage 4/5)
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

LLM_MODEL = "gemini-3-flash-preview"   # cheap & fast for tutorial
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
tavily = TavilySearch(max_results=3, search_depth="basic")


# ---------------------------------------------------------------------------
# 1. STATE - reducers now matter for real
# ---------------------------------------------------------------------------
# Reminder: Annotated[list, add] = "concatenate, don't overwrite."
# In Stage 6, multiple searcher nodes will be writing `findings`
# concurrently. Without `add`, we'd lose all but one of them.
class ResearchState(TypedDict):
    query: str
    sub_questions: Annotated[list[str], add]
    findings: Annotated[list[str], add]
    final_summary: str


# ---------------------------------------------------------------------------
# 2. PLANNER (same as before - Gemini + structured output)
# ---------------------------------------------------------------------------
class SubQuestions(BaseModel):
    """A decomposition of a research query into focused sub-questions."""
    questions: list[str] = Field(
        description=(
            "3 to 5 focused sub-questions that together would answer the "
            "user's original query. Each must be self-contained and "
            "answerable by a web search."
        )
    )

planner_llm = llm.with_structured_output(SubQuestions)

def planner_node(state: ResearchState) -> dict:
    query = state["query"]
    print(f"[planner] decomposing: {query!r}")
    prompt = (
        "You are a research planner. Decompose the following query into "
        "3-5 focused sub-questions, each answerable by a single web search.\n\n"
        f"Query: {query}"
    )
    result: SubQuestions = planner_llm.invoke(prompt)
    print(f"[planner] produced {len(result.questions)} sub-questions")
    return {"sub_questions": result.questions}


# ---------------------------------------------------------------------------
# 3. THE FAN-OUT FUNCTION (the heart of this stage)
# ---------------------------------------------------------------------------
# This is NOT a node - it's a ROUTING FUNCTION used as a conditional edge.
# Routing functions return either:
#   * a string (the name of the next node), or
#   * a list of strings (multiple next nodes), or
#   * a list of Send(...) objects (fan-out with custom inputs).
#
# We return Sends because we want each spawned searcher to receive a
# DIFFERENT input (a different sub-question), not the full state.
#
# Each Send has two parts:
#   Send(node_name, payload)
#     - node_name: the node to spawn
#     - payload  : a dict that becomes the INPUT STATE for that one
#                  spawned invocation. It does NOT have to match the
#                  global state schema - it can be a smaller, focused
#                  dict tailored to the worker.
#
# This is how we achieve CONTEXT ISOLATION: each searcher sees only
# {"sub_question": "..."} - not the rest of the world.
def dispatch_searchers(state: ResearchState) -> list[Send]:
    sub_qs = state["sub_questions"]
    print(f"[dispatcher] fanning out {len(sub_qs)} parallel searchers")
    return [
        Send("searcher", {"sub_question": sq})
        for sq in sub_qs
    ]


# ---------------------------------------------------------------------------
# 4. SEARCHER NODE - runs in parallel, sees only its own sub-question
# ---------------------------------------------------------------------------
# IMPORTANT: notice the type hint. The searcher's input is NOT the full
# ResearchState - it's a small dict with just `sub_question`. This is
# what the Send payload delivered.
#
# But the searcher's RETURN value IS merged back into the global state,
# and because `findings` has the `add` reducer, all parallel searchers'
# findings get concatenated safely.
class SearcherInput(TypedDict):
    sub_question: str

def searcher_node(state: SearcherInput) -> dict:
    sub_q = state["sub_question"]
    print(f"  [searcher] START   {sub_q[:60]!r}")

    raw = tavily.invoke({"query": sub_q})
    results = raw.get("results", [])
    findings = [
        f"Q: {sub_q}\n  -> [{r.get('title','?')}] {r.get('content','')[:180]}"
        for r in results
    ]
    print(f"  [searcher] DONE    {sub_q[:60]!r}  ({len(findings)} hits)")

    # Returning a list with the `add` reducer = "append these to the
    # global findings list." Other parallel searchers' returns will
    # likewise be appended; LangGraph handles ordering & merging.
    return {"findings": findings}


# ---------------------------------------------------------------------------
# 5. SUMMARIZER NODE - runs AFTER all parallel searchers finish
# ---------------------------------------------------------------------------
# How does it know to wait for all of them? Because of how we wire
# the edges below: the searchers all converge into the summarizer, and
# LangGraph automatically barriers (waits) until every parallel branch
# has completed before running the convergence node.
def summarizer_node(state: ResearchState) -> dict:
    findings = state["findings"]
    query = state["query"]
    print(f"[summarizer] composing answer from {len(findings)} findings")

    findings_block = "\n\n".join(findings) or "(no findings)"
    prompt = (
        "You are a research summarizer. Given the findings below from N "
        "parallel web searches, produce a concise (4-6 sentence) answer "
        "to the user's original query. Do not invent facts.\n\n"
        f"Original query: {query}\n\n"
        f"Findings:\n{findings_block}"
    )
    response = llm.invoke(prompt)
    return {"final_summary": response.content}


# ---------------------------------------------------------------------------
# 6. WIRE THE GRAPH (this is where Send shows up)
# ---------------------------------------------------------------------------
builder = StateGraph(ResearchState)
builder.add_node("planner",    planner_node)
builder.add_node("searcher",   searcher_node)
builder.add_node("summarizer", summarizer_node)

# Sequential entry: START -> planner
builder.add_edge(START, "planner")

# THE NEW PIECE: a CONDITIONAL EDGE.
# Args:
#   "planner"            -> the source node
#   dispatch_searchers   -> the routing function (returns a list of Sends)
#   ["searcher"]         -> the set of POSSIBLE next nodes (used for graph
#                           validation & visualization). Optional but
#                           recommended.
#
# `add_conditional_edges` is the API we use for ALL dynamic routing -
# whether returning a string, a list of strings, or a list of Sends.
# This is the same API we'll use in Stage 7 for branching.
builder.add_conditional_edges(
    "planner",
    dispatch_searchers,
    ["searcher"],
)

# All searchers converge here. LangGraph automatically waits for every
# parallel `searcher` invocation to finish before running `summarizer`.
builder.add_edge("searcher", "summarizer")
builder.add_edge("summarizer", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 7. RUN
# ---------------------------------------------------------------------------
# Watch the print order: you should see all [searcher] STARTs before
# any DONEs (they kick off concurrently), and the DONE messages will
# arrive in non-deterministic order depending on which Tavily call
# finishes first. That's the visible signal that fan-out is working.
if __name__ == "__main__":
    initial_state: ResearchState = {
        "query": "How does LangGraph compare to AutoGen and CrewAI for multi-agent systems?",
        "sub_questions": [],
        "findings": [],
        "final_summary": "",
    }
    final_state = graph.invoke(initial_state)

    print("\n=== FINAL STATE ===")
    print("query:", final_state["query"])
    print("\nsub_questions:")
    for q in final_state["sub_questions"]:
        print("  -", q)
    print(f"\nfindings: {len(final_state['findings'])} total")
    for f in final_state["findings"][:3]:
        print("  *", f[:140], "...")
    print("\nfinal_summary:")
    print(final_state["final_summary"])
