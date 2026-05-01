"""
Stage 1: The smallest possible LangGraph program.

Goal: Understand the 4 building blocks of every LangGraph app:
    1. State    - the shared "whiteboard" all nodes read/write
    2. Nodes    - plain functions that take state and return updates
    3. Edges    - rules that say "go to this node next"
    4. Compile  - turn the blueprint into a runnable graph
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE: the shape of the data that flows through our graph
# ---------------------------------------------------------------------------
# Think of this as the "whiteboard" every node can read from and write to.
# It's just a TypedDict - a regular Python dict with type hints.
# LangGraph uses this to know what fields exist and how to merge updates.
class HelloState(TypedDict):
    query: str       # what the user asked
    response: str    # what we'll answer


# ---------------------------------------------------------------------------
# 2. NODE: a plain Python function
# ---------------------------------------------------------------------------
# A node receives the full current state and returns a PARTIAL update.
# We do NOT mutate state directly - we return a dict of fields to merge.
# LangGraph handles the merging for us.
def echo_node(state: HelloState) -> dict:
    user_query = state["query"]
    answer = f"You asked: {user_query}"
    # Return only the fields we want to update. LangGraph merges this
    # into the existing state.
    return {"response": answer}


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH: wire nodes together with edges
# ---------------------------------------------------------------------------
# StateGraph(HelloState) tells LangGraph "the state on the whiteboard
# will follow this schema."
builder = StateGraph(HelloState)

# Register our node. The first arg is the node's NAME (a string we pick),
# the second is the function to run.
builder.add_node("echo", echo_node)

# Edges define the flow. START is a built-in marker meaning "the entry point".
# END is the built-in marker meaning "we're done".
# This says: START -> echo -> END
builder.add_edge(START, "echo")
builder.add_edge("echo", END)


# ---------------------------------------------------------------------------
# 4. COMPILE: turn the blueprint into a runnable graph
# ---------------------------------------------------------------------------
graph = builder.compile()


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # invoke() takes the INITIAL state and runs the graph to completion,
    # returning the FINAL state.
    initial_state = {"query": "What is the capital of France?", "response": ""}
    final_state = graph.invoke(initial_state)

    print("Initial state:", initial_state)
    print("Final state:  ", final_state)
