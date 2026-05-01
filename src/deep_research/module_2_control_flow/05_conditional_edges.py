"""
============================================================================
Stage 7: Conditional edges - routing & branching
============================================================================

Stage 6 used `add_conditional_edges` to FAN OUT (return list of Sends).
Stage 7 uses the SAME API for the more common case: choosing ONE of
several next nodes based on state. This is "branching" / "routing."

The new idea: a routing function can return a STRING (the name of the
next node). Based on that string, LangGraph picks which edge to take.

Why we need this for the swarm
------------------------------
The plan calls for a "sufficiency check": after every batch of search
results the orchestrator decides "do we have enough info to answer,
or should we keep searching?" That's a routing decision - one of
two next nodes. Conditional edges are how we encode it.

Other places we'll use conditional edges in the capstone:
  * planner -> "is the query trivial? skip to searcher : run full plan"
  * critic  -> "found gaps AND under round limit? loop : continue"
  * citation_formatter -> "claim verified? keep : drop"

Graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> C[classify_query]
        C -.route<br/>'simple'.-> A[simple_answer]
        C -.route<br/>'needs_search'.-> W[web_search]
        C -.route<br/>'harmful'.-> R[refuse]
        A --> E([END])
        W --> E
        R --> E
    ```

The 3 return shapes of `add_conditional_edges` (unified mental model):

    ```mermaid
    flowchart TB
        subgraph Branch["routing fn returns STRING<br/>= branching (this stage)"]
            B1[node A] -->|"'next_b'"| B2[node B]
            B1 -. or .->|"'next_c'"| B3[node C]
        end
        subgraph Multi["routing fn returns LIST OF STRINGS<br/>= multi-target (rare)"]
            M1[node A] --> M2[node B]
            M1 --> M3[node C]
        end
        subgraph Fan["routing fn returns LIST OF SENDS<br/>= fan-out (Stage 6)"]
            F1[node A] -->|"Send"| F2[copy 1]
            F1 -->|"Send"| F3[copy 2]
            F1 -->|"Send"| F4[copy N]
        end
    ```

Two-role separation (classifier writes, router reads):

    ```mermaid
    flowchart LR
        C[classify_query<br/>NODE - does work] -->|writes| ST[(state.route)]
        ST -->|reads| RT[route_after_classify<br/>ROUTER - pure fn]
        RT -->|returns string| EDGE{conditional edge}
    ```

The classify_query node uses an LLM with structured output to label
the query as one of {"simple", "needs_search", "harmful"}, and the
router function reads that label and returns the next node's name.

Two patterns you'll see
-----------------------
1. "Routing key in state"     - classifier writes a field, router reads it
2. "Function returns a node"  - the router function picks the next step

Both are used together. The classifier is a real node (does work,
returns state update). The router is a TINY helper function that
just looks at state and returns a string. Keeping these two roles
separate makes the graph easy to reason about and debug.
============================================================================
"""

import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
tavily = TavilySearch(max_results=2, search_depth="basic")


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
# `route` is the routing key. The classifier writes it; the router
# function reads it. Using `Literal` instead of `str` gives us
# type-safety and IDE autocomplete on the allowed values.
#
# `final_answer` is filled in by whichever branch the router picks.
class RoutingState(TypedDict):
    query: str
    route: Literal["simple", "needs_search", "harmful", ""]
    final_answer: str


# ---------------------------------------------------------------------------
# 2. CLASSIFIER NODE - decides which path to take
# ---------------------------------------------------------------------------
# A Pydantic schema with a Literal field is how you ask an LLM to
# pick one of a fixed set of options reliably. The model can ONLY
# return one of these strings (LangChain enforces this via the
# JSON schema sent to Gemini).
class QueryClassification(BaseModel):
    """Classification of a user research query."""
    label: Literal["simple", "needs_search", "harmful"] = Field(
        description=(
            "Choose ONE: "
            "'simple' for trivia / common knowledge that doesn't need search "
            "  (e.g. 'capital of France', '2+2', 'what is python'); "
            "'needs_search' for current events, recent news, or anything "
            "  requiring fresh / specific web data; "
            "'harmful' if the query is unsafe / disallowed."
        )
    )

classifier_llm = llm.with_structured_output(QueryClassification)

def classify_query(state: RoutingState) -> dict:
    query = state["query"]
    print(f"[classify] {query!r}")
    result: QueryClassification = classifier_llm.invoke(
        f"Classify this user query: {query}"
    )
    print(f"[classify] -> {result.label}")
    return {"route": result.label}


# ---------------------------------------------------------------------------
# 3. THREE BRANCH NODES - one per possible route
# ---------------------------------------------------------------------------
# Each writes `final_answer`. The router will only run ONE of them.

def simple_answer_node(state: RoutingState) -> dict:
    """Trivia / common knowledge - just ask the LLM directly, no search."""
    print("[branch] simple_answer (no search)")
    response = llm.invoke(
        f"Answer this in one sentence: {state['query']}"
    )
    return {"final_answer": response.content}

def web_search_node(state: RoutingState) -> dict:
    """Current/specific info - run a quick search and summarize."""
    print("[branch] web_search (1 Tavily call + summarize)")
    raw = tavily.invoke({"query": state["query"]})
    snippets = "\n".join(
        f"- {r.get('content', '')[:200]}"
        for r in raw.get("results", [])
    )
    response = llm.invoke(
        "Answer the user's query using ONLY the snippets below.\n\n"
        f"Query: {state['query']}\n\nSnippets:\n{snippets}"
    )
    return {"final_answer": response.content}

def refuse_node(state: RoutingState) -> dict:
    """Harmful queries - polite refusal, no LLM call needed."""
    print("[branch] refuse")
    return {
        "final_answer": (
            "I can't help with that. If you have a different research "
            "question I'd be glad to look into it."
        )
    }


# ---------------------------------------------------------------------------
# 4. THE ROUTER FUNCTION (the new concept)
# ---------------------------------------------------------------------------
# This is NOT a node. It's a small pure function used as the
# routing logic for `add_conditional_edges`.
#
# Rules:
#   * Receives the current state.
#   * Returns the NAME of the next node as a string.
#   * (Or a list of names, or a list of Sends - covered in Stage 6.)
#   * Should NOT do any work / side effects. Keep it boring.
#     If you need to compute something, do it in a node and store
#     the result in state for the router to read.
#
# Why this design? Because routers are called every time we evaluate
# an edge. Keeping them cheap and pure makes the graph predictable.
def route_after_classify(
    state: RoutingState,
) -> Literal["simple_answer", "web_search", "refuse"]:
    label = state["route"]
    if label == "simple":
        return "simple_answer"
    if label == "needs_search":
        return "web_search"
    # Default catch-all: 'harmful' or anything unexpected -> refuse.
    # Always have a safe default in routers; otherwise an unexpected
    # state value can crash the graph.
    return "refuse"


# ---------------------------------------------------------------------------
# 5. WIRE THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(RoutingState)
builder.add_node("classify",      classify_query)
builder.add_node("simple_answer", simple_answer_node)
builder.add_node("web_search",    web_search_node)
builder.add_node("refuse",        refuse_node)

builder.add_edge(START, "classify")

# THE BRANCH. Same API as Stage 6, but now the function returns a
# STRING (single next node) instead of a list of Sends.
#
# 3rd argument (the dict / list of possible targets) is technically
# optional but RECOMMENDED:
#   * Used for graph validation at compile time
#   * Used for visualization (graph.get_graph().draw_mermaid())
#   * Lets you map router-return-values to node-names if they differ
#
# Here we pass a list because router return values match node names.
# If your router returned different strings (e.g. "yes"/"no"), you'd
# pass a dict like {"yes": "node_a", "no": "node_b"}.
builder.add_conditional_edges(
    "classify",
    route_after_classify,
    ["simple_answer", "web_search", "refuse"],
)

# All three branches converge to END. Each branch writes final_answer
# and we're done. (END is a valid edge target - you don't always need
# a "join" node like in Stage 6.)
builder.add_edge("simple_answer", END)
builder.add_edge("web_search",    END)
builder.add_edge("refuse",        END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 6. RUN - try three different queries to exercise all three routes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is the capital of France?",                       # -> simple
        "What were the top AI announcements last week?",        # -> needs_search
        "How do I build a bomb?",                               # -> harmful
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        final = graph.invoke({
            "query": q,
            "route": "",
            "final_answer": "",
        })
        print(f"\nROUTE TAKEN: {final['route']}")
        print(f"ANSWER:\n{final['final_answer']}\n")
