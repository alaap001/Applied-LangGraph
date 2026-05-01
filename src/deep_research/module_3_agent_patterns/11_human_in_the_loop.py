"""
============================================================================
Stage 13: Human-in-the-loop, checkpoints & time-travel
============================================================================

Three powerful capabilities that all rely on ONE underlying mechanism
(checkpointing). After this stage, you'll see why every "production"
LangGraph app passes a checkpointer at compile time.

What this stage teaches
-----------------------
1. CHECKPOINTING (the foundation)
   - `MemorySaver` (in-process) and friends save the full graph state
     after every step, indexed by a `thread_id`.
   - Same thread_id == same conversation. Resume, replay, branch.
   - Required for all of (2), (3), (4).

2. INTERRUPT (human-in-the-loop)
   - Inside a node, call `interrupt(payload)`. The graph PAUSES.
   - The caller sees the payload (e.g. "approve this $5 spend?").
   - To resume, call `graph.invoke(Command(resume=value), config=...)`
     with the same thread_id; that `value` becomes the return value
     of `interrupt()`.

3. TIME TRAVEL
   - `graph.get_state_history(config)` returns every checkpoint.
   - Pass an OLD checkpoint config back into `.invoke()` and the graph
     re-runs from that point - perfect for debugging or "redo with a
     small change."

4. STATE EDITING
   - `graph.update_state(config, {"foo": "bar"})` lets you splice
     values into past state and resume from there. Combined with
     time-travel this is the "rewind & rewrite" feature in the
     PROJECT2_PLAN UI.

Why this is non-negotiable for the capstone
-------------------------------------------
* Resume-after-crash: the swarm makes 20+ LLM calls. If call 19 fails
  on a transient API error, you DO NOT want to redo calls 1-18.
* Cost approvals: "you're about to spend $4.20 on this query, OK?"
  is exactly an `interrupt()` use case.
* Replays for evals: feed the same thread_id+initial state into the
  graph, get bit-for-bit identical traces.
* Killer UI: the "rewind to checkpoint N and edit the plan" feature
  in PROJECT2_PLAN section 10 is `get_state_history` + `update_state`.

Graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> P[propose_search]
        P --> RA[request_approval<br/>calls interrupt]
        RA -.route<br/>'ok'.-> RUN[run_search<br/>full Tavily]
        RA -.route<br/>'cheaper'.-> CHEAP[run_cheap_search<br/>1-result]
        RA -.route<br/>'cancel'.-> CAN[cancelled]
        RUN --> E([END])
        CHEAP --> E
        CAN --> E
    ```

Interrupt + resume call sequence:

    ```mermaid
    sequenceDiagram
        participant U as User
        participant G as Graph
        participant CP as Checkpointer
        U->>G: invoke(initial, config={thread_id})
        G->>G: propose_search runs
        G->>G: request_approval calls interrupt({...})
        G->>CP: save state @ checkpoint K
        G-->>U: yield {__interrupt__: payload}
        Note over U: human reads payload<br/>decides 'ok' / 'cheaper' / 'cancel'
        U->>G: invoke(Command(resume='ok'), config={thread_id})
        G->>CP: load state @ checkpoint K
        G->>G: interrupt() returns 'ok'
        G->>G: route to run_search
        G-->>U: final state
    ```

Time travel with `get_state_history`:

    ```mermaid
    flowchart TB
        H["graph.get_state_history(config)<br/>returns list, newest-first"] --> CP1["checkpoint 0<br/>(START)"]
        H --> CP2["checkpoint 1<br/>(after propose)"]
        H --> CP3["checkpoint 2<br/>(at interrupt)"]
        H --> CP4["checkpoint 3<br/>(after run_search)"]
        H --> CP5["checkpoint 4<br/>(END)"]
        CP3 -->|"invoke(Command(resume='cheaper'),<br/>config=cp3.config)"| BR[branch:<br/>NEW timeline<br/>cheaper instead]
    ```
============================================================================
"""

import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
# NEW: the checkpointer + the interrupt primitive.
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tavily_basic = TavilySearch(max_results=3, search_depth="basic")
tavily_cheap = TavilySearch(max_results=1, search_depth="basic")  # "cheaper"


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
class AssistantState(TypedDict):
    query: str
    proposed_search: str
    estimated_cost_usd: float
    approval: Literal["ok", "cheaper", "cancel", ""]
    final_answer: str


# ---------------------------------------------------------------------------
# 2. PROPOSE NODE - LLM drafts the search query
# ---------------------------------------------------------------------------
class SearchPlan(BaseModel):
    refined_query: str = Field(description="A focused web-search query (3-10 words).")

planner_llm = llm.with_structured_output(SearchPlan)

def propose_search(state: AssistantState) -> dict:
    print(f"[propose] drafting a search plan for: {state['query']!r}")
    plan = planner_llm.invoke(
        f"Refine this into a focused web-search query (3-10 words): "
        f"{state['query']}"
    )
    # Pretend cost. In real life you'd estimate from token counts.
    return {
        "proposed_search": plan.refined_query,
        "estimated_cost_usd": 0.05,
    }


# ---------------------------------------------------------------------------
# 3. APPROVAL NODE - this is where `interrupt()` happens
# ---------------------------------------------------------------------------
# Inside a node, calling `interrupt(payload)` does TWO things:
#   1. PAUSES the graph (control returns to the caller of .invoke /.stream).
#   2. The payload becomes part of the special "__interrupt__" event the
#      caller receives. This is where you put info the human needs to
#      decide ("plan: ..., cost: $0.05").
#
# When you resume with Command(resume=X), the X value is what
# interrupt() RETURNS (yes, returns - resuming makes the call site
# come back to life and continue). This is why we capture it into a
# variable and use it.
def request_approval(state: AssistantState) -> dict:
    print(f"[approval] pausing for human input (interrupt)")

    # When resumed, `human_decision` will be whatever was passed
    # in Command(resume=...) - a string here, but it could be any
    # JSON-serializable value (dicts work great for richer feedback).
    human_decision = interrupt({
        "kind": "approval_request",
        "search_plan": state["proposed_search"],
        "estimated_cost_usd": state["estimated_cost_usd"],
        "options": ["ok", "cheaper", "cancel"],
        "message": (
            f"About to run: {state['proposed_search']!r}. "
            f"Estimated ${state['estimated_cost_usd']:.2f}. "
            f"Reply with 'ok' / 'cheaper' / 'cancel'."
        ),
    })

    print(f"[approval] resumed with decision: {human_decision!r}")
    # Whatever the human chose flows into state for the router below.
    return {"approval": human_decision}


# ---------------------------------------------------------------------------
# 4. ROUTER - look at approval, pick the next node
# ---------------------------------------------------------------------------
# Standard Stage 7 conditional edge. The interesting bit was step 3.
def route_after_approval(
    state: AssistantState,
) -> Literal["run_search", "run_cheap_search", "cancelled"]:
    a = state["approval"]
    if a == "ok":      return "run_search"
    if a == "cheaper": return "run_cheap_search"
    return "cancelled"


# ---------------------------------------------------------------------------
# 5. WORK NODES
# ---------------------------------------------------------------------------
def run_search(state: AssistantState) -> dict:
    print("[work] running full Tavily search")
    raw = tavily_basic.invoke({"query": state["proposed_search"]})
    snippets = "\n".join(
        f"- {r.get('content','')[:200]}" for r in raw.get("results", [])
    )
    answer = llm.invoke(
        "Answer using only these snippets.\n\n"
        f"Q: {state['query']}\nSnippets:\n{snippets}"
    ).content
    return {"final_answer": answer}

def run_cheap_search(state: AssistantState) -> dict:
    print("[work] running 1-result Tavily search (cheap)")
    raw = tavily_cheap.invoke({"query": state["proposed_search"]})
    snippets = "\n".join(
        f"- {r.get('content','')[:200]}" for r in raw.get("results", [])
    )
    answer = llm.invoke(
        f"Answer briefly.\nQ: {state['query']}\nSnippets:\n{snippets}"
    ).content
    return {"final_answer": answer}

def cancelled_node(state: AssistantState) -> dict:
    print("[work] user cancelled")
    return {"final_answer": "(cancelled by user)"}


# ---------------------------------------------------------------------------
# 6. WIRE + COMPILE WITH CHECKPOINTER
# ---------------------------------------------------------------------------
# This is the line that enables EVERYTHING in this stage:
#
#     graph = builder.compile(checkpointer=MemorySaver())
#
# Without it, interrupt() can't pause (no place to save state to),
# get_state_history can't show you anything, update_state can't
# rewrite history.
#
# MemorySaver is in-process (lost on restart - dev only). Production
# uses SqliteSaver / AsyncPostgresSaver - SAME API, persistent.
builder = StateGraph(AssistantState)
builder.add_node("propose_search",    propose_search)
builder.add_node("request_approval",  request_approval)
builder.add_node("run_search",        run_search)
builder.add_node("run_cheap_search",  run_cheap_search)
builder.add_node("cancelled",         cancelled_node)

builder.add_edge(START, "propose_search")
builder.add_edge("propose_search", "request_approval")
builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    ["run_search", "run_cheap_search", "cancelled"],
)
builder.add_edge("run_search",       END)
builder.add_edge("run_cheap_search", END)
builder.add_edge("cancelled",        END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 7. DEMO
# ---------------------------------------------------------------------------
def demo_interrupt_flow():
    """Run the graph until interrupt, then resume with each of the
    three possible decisions on its own thread."""
    print("\n=== Demo: interrupt + resume on three threads ===")

    initial = {
        "query": "Latest stable Python version and its biggest new feature",
        "proposed_search": "",
        "estimated_cost_usd": 0.0,
        "approval": "",
        "final_answer": "",
    }

    for decision in ["ok", "cheaper", "cancel"]:
        print(f"\n--- thread for decision={decision!r} ---")
        # The thread_id is how the checkpointer knows "same conversation"
        # across multiple .invoke calls. We use a different id per demo.
        config = {"configurable": {"thread_id": f"demo-{decision}"}}

        # First .invoke runs until the graph hits interrupt().
        # The returned value is a dict containing __interrupt__ events.
        first = graph.invoke(initial, config=config)

        # Inspect the interrupt payload (this is what the UI would show
        # to the human).
        interrupts = first.get("__interrupt__", [])
        if interrupts:
            print(f"[caller] graph paused. payload:")
            for it in interrupts:
                print(f"         {it.value['message']}")

        # Second .invoke RESUMES the same thread by passing
        # Command(resume=value). The graph picks up exactly where it
        # left off; interrupt() returns `value` inside the node.
        final = graph.invoke(Command(resume=decision), config=config)
        print(f"[caller] final answer: {final['final_answer'][:140]!r}")


def demo_time_travel():
    """Show the checkpoint history of one of the threads, then re-run
    from an earlier checkpoint with a state edit."""
    print("\n=== Demo: time-travel + state editing ===")

    # We'll reuse the 'ok' thread from the first demo (so we already
    # have a finished run on it).
    config = {"configurable": {"thread_id": "demo-ok"}}

    # get_state_history returns checkpoints newest-first. Each has:
    #   .values  -> the state dict at that point
    #   .next    -> tuple of node names that would run next
    #   .config  -> a config dict you can pass back to .invoke to
    #               REPLAY from this exact checkpoint
    history = list(graph.get_state_history(config))
    print(f"thread has {len(history)} checkpoints. summary:")
    for i, snap in enumerate(history):
        next_node = snap.next[0] if snap.next else "<end>"
        approval = snap.values.get("approval", "-")
        print(f"  [{i}] next={next_node:<20} approval={approval!r}")

    # Find the checkpoint just before the approval was set (i.e. when
    # request_approval is the next node to run). We'll rewrite the
    # approval there and replay.
    target = next(
        (s for s in history
         if s.next and "request_approval" in s.next
         and s.values.get("approval") == ""),
        None,
    )
    if target is None:
        print("  (couldn't find a pre-approval checkpoint to rewind to)")
        return

    print(f"\nrewinding to a pre-approval checkpoint and forcing 'cheaper'")

    # Resume from that historical config with a different decision.
    # Note we don't even need update_state here - Command(resume=...)
    # plus the historical config gives us a "redo with a different
    # answer" behavior.
    final = graph.invoke(Command(resume="cheaper"), config=target.config)
    print(f"[caller] new final: {final['final_answer'][:140]!r}")


if __name__ == "__main__":
    demo_interrupt_flow()
    demo_time_travel()
