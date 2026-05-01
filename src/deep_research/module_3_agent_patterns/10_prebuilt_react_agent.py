"""
============================================================================
Stage 12: The prebuilt ReAct agent (and when NOT to use it)
============================================================================

In Stage 11 we built ReAct from scratch: agent_node + tools_node + a
should_continue router, ~80 lines including comments. LangGraph ships a
one-liner that does all of that for you:

    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(model=llm, tools=TOOLS)

The whole agent. One line. So why did we just spend a whole stage
building it ourselves?

Because the prebuilt is GREAT for 80% of cases and a TRAP for 20%.
Knowing which is which is the entire point of this stage.

What `create_react_agent` gives you for free
--------------------------------------------
1. The same 2-node ReAct cycle from Stage 11 (agent <-> tools).
2. A production-grade ToolNode that:
     - Runs multiple tool_calls in parallel (we did sequentially)
     - Catches and reports tool exceptions back to the LLM
     - Handles streaming
3. Built-in support for:
     - System prompts (`prompt=` param)
     - State schema extension (extra fields beyond just `messages`)
     - Pre/post-model hooks (run code before/after each LLM call)
     - Structured output as the FINAL answer (`response_format=`)
     - Checkpointer + memory store passthrough (Stage 13/19 territory)
4. A clean, typed graph object you can wire into a parent graph as a
   subgraph - exactly the Stage 9 pattern.

When to USE the prebuilt
------------------------
* You have a single agent with a few tools and you just want it to work
* You want a clean, opinionated default
* You're going to wire several agents together and want each one
  written consistently (less code, fewer footguns)
* You don't need to customize the loop's internals

When to AVOID it / drop down to hand-rolled
-------------------------------------------
* You need a 3+ node loop (agent -> validator -> tools -> agent)
* You need Send-based parallel fan-out INSIDE the agent (Stage 6)
* You want to enforce hard tool-call caps in code, not just trust
  the LLM ("max 8 tool calls per agent" from PROJECT2_PLAN section 7)
* You need state-shape transformations between LLM calls
* You want tracing/observability hooks at points the prebuilt doesn't expose

Graph topology (mermaid) - same as Stage 11 internally:

    ```mermaid
    flowchart LR
        S([START]) --> A[agent]
        A -.tool_calls.-> T[tools<br/>parallel execution<br/>+ error handling]
        T --> A
        A -.no tool_calls.-> E([END])
    ```

Pre/post-model hooks (where kill-switches live):

    ```mermaid
    flowchart LR
        IN[input messages] --> PRE[pre_model_hook<br/>inject context,<br/>enforce caps]
        PRE --> LLM[LLM call]
        LLM --> POST[post_model_hook<br/>count tokens,<br/>log events]
        POST --> OUT[next step]
    ```

`response_format=` adds a final coercion step:

    ```mermaid
    flowchart LR
        A[agent loop<br/>tool_calls cycle] -->|"LLM stops"| FINAL[final AI message]
        FINAL --> SO[extra LLM call<br/>with_structured_output]
        SO --> RES["state.structured_response<br/>= Pydantic instance"]
    ```

Capstone agent split (which agents are prebuilt vs hand-rolled):

    ```mermaid
    flowchart TB
        subgraph Prebuilt["create_react_agent<br/>(simple loop)"]
            S1[Searcher]
            S2[Browser]
            S3[Fact-Checker<br/>+ response_format]
        end
        subgraph Hand["Hand-rolled<br/>(custom mechanics)"]
            H1[Critic<br/>uses Command cycles]
            H2[Lead Orchestrator<br/>Send fan-out + sufficiency]
            H3[Synthesizer<br/>structured-output only,<br/>no tools]
        end
    ```

For the capstone we'll use the prebuilt for the SEARCHER and
FACT_CHECKER (simple agents), and hand-roll the LEAD ORCHESTRATOR
(complex - needs sufficiency check, kill switches, fan-out logic).

What this file demonstrates
---------------------------
1. Tutorial A: simplest prebuilt agent (one line setup)
2. Tutorial B: prebuilt agent with a SYSTEM PROMPT and pre/post hooks
3. Tutorial C: prebuilt agent with STRUCTURED FINAL OUTPUT
4. Tutorial D: drop the prebuilt agent into a PARENT GRAPH as a subgraph
   (the capstone pattern - this is how the swarm uses agents)
============================================================================
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
# THE star of this stage:
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ---------------------------------------------------------------------------
# 1. TOOLS - same shape as Stage 11
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information.

    Use this when the user asks about recent events, current facts,
    or anything that requires fresh online data. Returns short snippets.
    """
    tavily = TavilySearch(max_results=3, search_depth="basic")
    raw = tavily.invoke({"query": query})
    return "\n".join(
        f"- ({r.get('title','?')}) {r.get('content','')[:200]}"
        for r in raw.get("results", [])
    ) or "(no results)"


@tool
def get_current_time() -> str:
    """Get the current date and time in ISO format."""
    return datetime.now().isoformat()


TOOLS = [web_search, get_current_time]


# ===========================================================================
# TUTORIAL A: the one-liner
# ===========================================================================
# Compare this to the ~80 lines in Stage 11. Same behavior, same
# graph shape, more battle-tested code. Use this for any "simple
# tool-using agent" need.
def demo_a_minimal():
    print("\n=== Tutorial A: minimal prebuilt agent ===")

    agent = create_react_agent(model=llm, tools=TOOLS)

    # The agent's input shape is `MessagesState` ({"messages": [...]})
    # exactly like our hand-rolled version. .invoke returns the final
    # state; the assistant's answer is the last message's content.
    out = agent.invoke({"messages": [("user", "What's today's date?")]})
    print("ANSWER:", out["messages"][-1].content)


# ===========================================================================
# TUTORIAL B: system prompt + a "post_model_hook" for token accounting
# ===========================================================================
# Two things this teaches:
#
#   1. `prompt=` lets you set a SYSTEM PROMPT once at agent creation.
#      Don't pass system messages in the input every turn - that
#      pollutes message history. Use this instead.
#
#   2. `pre_model_hook=` and `post_model_hook=` are functions called
#      RIGHT BEFORE and RIGHT AFTER every LLM call. Use them to:
#        - count tokens / cost
#        - inject extra context (e.g. recent memory hits)
#        - enforce hard limits (raise an exception to stop)
#        - log custom events for streaming UIs
#
# These hooks are where you implement the "max_tool_calls_per_agent = 8"
# kill-switch from PROJECT2_PLAN section 7. The prebuilt agent gives
# you the hook points; you fill in the policy.
def demo_b_prompt_and_hooks():
    print("\n=== Tutorial B: system prompt + post-call hook ===")

    SYSTEM_PROMPT = (
        "You are a careful research assistant. When uncertain, USE THE "
        "web_search TOOL. Always cite source titles inline like (TitleHere)."
    )

    # The hook receives the FULL state and returns a state update dict.
    # Returning {} means "no changes." We use it here just to log.
    call_count = {"n": 0}
    def post_model_hook(state):
        call_count["n"] += 1
        last = state["messages"][-1]
        kind = "tool_calls" if last.tool_calls else "answer"
        print(f"  [post-hook] LLM call #{call_count['n']} -> {kind}")
        return {}

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        post_model_hook=post_model_hook,
    )

    out = agent.invoke({"messages": [
        ("user", "Briefly: what is the latest version of Python released?"),
    ]})
    print("ANSWER:", out["messages"][-1].content[:200], "...")


# ===========================================================================
# TUTORIAL C: structured final output via response_format
# ===========================================================================
# So far the agent's final answer is unstructured text. For programmatic
# downstream consumers (e.g. the synthesizer reading per-claim trust
# scores), you want STRUCTURED output - the same Stage 4 pattern.
#
# `response_format=PydanticModel` tells the prebuilt to:
#   1. Run the normal ReAct loop until the LLM stops calling tools.
#   2. Then make ONE more LLM call to coerce the final answer into the
#      Pydantic shape (using with_structured_output under the hood).
#   3. Put the parsed object in state under `structured_response`.
class SearchVerdict(BaseModel):
    """A focused answer to a research question with confidence + sources."""
    answer: str = Field(description="A 1-3 sentence answer to the user's question.")
    confidence: int = Field(ge=0, le=100, description="Confidence 0-100.")
    sources: list[str] = Field(
        default_factory=list,
        description="Source titles cited in the answer.",
    )

def demo_c_structured_output():
    print("\n=== Tutorial C: structured final output ===")

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt="You are a research assistant. Use web_search when uncertain.",
        response_format=SearchVerdict,
    )

    out = agent.invoke({"messages": [
        ("user", "What are the top 2 open-source vector databases in 2025?"),
    ]})

    # The parsed Pydantic instance lives in state["structured_response"].
    verdict: SearchVerdict = out["structured_response"]
    print(f"  answer:     {verdict.answer}")
    print(f"  confidence: {verdict.confidence}")
    print(f"  sources:    {verdict.sources}")


# ===========================================================================
# TUTORIAL D: prebuilt agent AS A SUBGRAPH inside a parent graph
# ===========================================================================
# This is THE pattern the capstone uses. The compiled prebuilt agent IS
# a runnable Pregel object - same as anything we've built. So we drop
# it into a parent graph as a single node.
#
# The parent graph here is intentionally tiny (planner -> agent -> done)
# but it shows the key idea: each "agent" in the swarm becomes a single
# node in the master graph, with internal complexity hidden.
def demo_d_agent_as_subgraph():
    print("\n=== Tutorial D: prebuilt agent embedded in a parent graph ===")

    # Step 1: build the prebuilt searcher agent (same as Tutorial A).
    searcher_agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt="You are a focused web researcher. Cite source titles inline.",
    )

    # Step 2: build a tiny parent graph that uses it.
    # Parent state has its OWN shape - we'll translate at the boundary.
    from typing import TypedDict
    class ParentState(TypedDict):
        question: str
        answer: str

    def search_using_agent(state: ParentState) -> dict:
        # Translate parent state -> agent input
        agent_input = {"messages": [("user", state["question"])]}
        # Run the agent - it's a Pregel, so .invoke works exactly like
        # any compiled graph.
        agent_out = searcher_agent.invoke(agent_input)
        # Translate agent output -> parent state update
        final_text = agent_out["messages"][-1].content
        return {"answer": final_text}

    parent = StateGraph(ParentState)
    parent.add_node("search", search_using_agent)
    parent.add_edge(START, "search")
    parent.add_edge("search", END)

    parent_graph = parent.compile()

    out = parent_graph.invoke({
        "question": "What's the difference between Qdrant and Pinecone in 2025?",
        "answer": "",
    })
    print("ANSWER:", out["answer"][:300], "...")


# ---------------------------------------------------------------------------
# Decision flowchart you should internalize
# ---------------------------------------------------------------------------
#   Need a tool-using agent?
#     - Custom 3+ node loop?            -> hand-roll (Stage 11 pattern)
#     - Need Send-based fan-out inside? -> hand-roll
#     - Need state-shape transforms?    -> hand-roll
#     - Otherwise                        -> create_react_agent (Tutorial A-C)
#
# In the capstone:
#   Searcher       -> create_react_agent (one tool, simple loop) [prebuilt]
#   Browser        -> create_react_agent                          [prebuilt]
#   Fact-Checker   -> create_react_agent + structured output      [prebuilt]
#   Critic         -> hand-roll (uses Command + cycles, Stage 8)
#   Orchestrator   -> hand-roll (Send fan-out + sufficiency check)
#   Synthesizer    -> hand-roll (no tools, just structured output)
#
# Roughly half-and-half. The prebuilt isn't a "use it for everything"
# tool; it's a "use it where it fits" tool.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    demo_a_minimal()
    demo_b_prompt_and_hooks()
    demo_c_structured_output()
    demo_d_agent_as_subgraph()
