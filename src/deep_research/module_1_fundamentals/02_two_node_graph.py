"""
============================================================================
Stage 2 + Stage 3 combined: Reducers + multi-node graphs
============================================================================

Stage 2: REDUCERS
    - How do we let nodes APPEND to a list instead of OVERWRITING it?
    - Answer: Annotated[list, add]  -> tells LangGraph "merge by concatenating"

Stage 3: MULTIPLE NODES
    - Wire two nodes in sequence: planner -> summarizer
    - The planner writes sub_questions; the summarizer reads them.
    - This is the same shape as the real Planner -> Orchestrator handoff
      we'll build later, just without the LLM.

Graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> P[planner_node]
        P --> SU[summarizer_node]
        SU --> E([END])
    ```

What a reducer does (concept diagram):

    ```mermaid
    flowchart TB
        subgraph NoReducer["WITHOUT reducer<br/>(default = overwrite)"]
            A1["state.findings = ['x']"]
            A2["node returns {findings: ['y']}"]
            A3["state.findings = ['y']<br/>'x' is LOST"]
            A1 --> A2 --> A3
        end
        subgraph WithReducer["WITH Annotated[list, add]<br/>(concatenate)"]
            B1["state.findings = ['x']"]
            B2["node returns {findings: ['y']}"]
            B3["state.findings = ['x', 'y']<br/>both kept"]
            B1 --> B2 --> B3
        end
    ```

Field-level merge strategy (which fields use which reducer):

    ```mermaid
    flowchart LR
        subgraph State[ResearchState]
            Q["query: str<br/>(overwrite)"]
            SQ["sub_questions: list<br/>(add = concat)"]
            F["findings: list<br/>(add = concat)"]
            FS["final_summary: str<br/>(overwrite)"]
        end
    ```
============================================================================
"""

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE: now with a LIST field that needs special merging behavior
# ---------------------------------------------------------------------------
# Look closely at `sub_questions`. We've wrapped its type in `Annotated[..., add]`.
#
# Without the reducer:
#   - If two nodes both return {"sub_questions": [...]}, the SECOND one
#     OVERWRITES the first. The first node's work is lost.
#
# With `Annotated[list[str], add]`:
#   - LangGraph sees the `add` reducer and uses it to MERGE updates.
#   - `add` here is `operator.add` from Python's stdlib, which on lists
#     means concatenation: [1,2] + [3,4] == [1,2,3,4].
#   - So if two nodes return sub_questions in parallel, both get appended.
#
# This is the single most important pattern for parallel agents.
# Every list/dict field that multiple nodes touch needs a reducer.
class ResearchState(TypedDict):
    query: str
    sub_questions: Annotated[list[str], add]    # APPEND, don't overwrite
    findings: Annotated[list[str], add]         # APPEND, don't overwrite
    final_summary: str                          # plain string, last write wins


# ---------------------------------------------------------------------------
# 2. NODE A: the "planner"
# ---------------------------------------------------------------------------
# Takes the user query and decomposes it into sub-questions.
# In the real project this will be an LLM call with structured output.
# For now: hardcoded logic so we focus on graph mechanics.
def planner_node(state: ResearchState) -> dict:
    query = state["query"]
    print(f"[planner] decomposing query: {query!r}")

    # Pretend the LLM gave us back 3 sub-questions.
    sub_qs = [
        f"What is the definition of: {query}?",
        f"What are the key examples of: {query}?",
        f"What are common misconceptions about: {query}?",
    ]

    # Notice: we return a list. Because the field has the `add` reducer,
    # LangGraph will CONCATENATE this with whatever was already there
    # (here, an empty list, so the result is just our 3 items).
    return {"sub_questions": sub_qs}


# ---------------------------------------------------------------------------
# 3. NODE B: the "summarizer"
# ---------------------------------------------------------------------------
# Reads sub_questions written by the planner and produces fake "findings"
# plus a final summary. Demonstrates that downstream nodes can READ what
# upstream nodes wrote.
def summarizer_node(state: ResearchState) -> dict:
    sub_qs = state["sub_questions"]
    print(f"[summarizer] received {len(sub_qs)} sub-questions")

    # Pretend we ran a search for each sub-question. One fake finding each.
    fake_findings = [f"Finding for: {q}" for q in sub_qs]

    summary = (
        f"Researched {len(sub_qs)} sub-questions for query "
        f"'{state['query']}'. Top finding: {fake_findings[0]}"
    )

    # Two updates in one return - LangGraph merges both into state.
    # `findings` uses the `add` reducer (concatenated).
    # `final_summary` has no reducer (overwritten / last-write-wins).
    return {
        "findings": fake_findings,
        "final_summary": summary,
    }


# ---------------------------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(ResearchState)

# Two nodes registered under names of our choosing.
builder.add_node("planner", planner_node)
builder.add_node("summarizer", summarizer_node)

# Edges: START -> planner -> summarizer -> END
# This is sequential. The summarizer cannot start until the planner
# has finished, because it depends on planner's output.
builder.add_edge(START, "planner")
builder.add_edge("planner", "summarizer")
builder.add_edge("summarizer", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANT: every field declared in the TypedDict must be present
    # in the initial state (TypedDicts have no defaults).
    # For the list fields, we start with [].
    initial_state: ResearchState = {
        "query": "vector databases",
        "sub_questions": [],
        "findings": [],
        "final_summary": "",
    }

    final_state = graph.invoke(initial_state)

    print("\n--- FINAL STATE ---")
    print("query:         ", final_state["query"])
    print("sub_questions: ")
    for q in final_state["sub_questions"]:
        print("  -", q)
    print("findings:      ")
    for f in final_state["findings"]:
        print("  -", f)
    print("final_summary: ", final_state["final_summary"])
