"""
============================================================================
Stage 11: Build the ReAct agent loop FROM SCRATCH
============================================================================

ReAct = "Reasoning + Acting." It's the canonical pattern behind every
"agent" you've ever used (ChatGPT with browsing, Claude with tools,
the Searcher in our capstone). The loop is dead simple:

    1. LLM reads the conversation.
    2. LLM either:
         (a) emits a "tool_call" (e.g. "call tavily_search('foo')"), or
         (b) emits a final text answer.
    3. If (a): we EXECUTE the tool, append the result to the
       conversation, and loop back to step 1.
    4. If (b): we stop and return the answer.

That's it. Most agent libraries are wrappers around this 4-step loop.
We're going to build it ourselves before using the prebuilt version
(Stage 12), so the prebuilt one stops feeling like magic.

Why "from scratch" matters
--------------------------
Once you've built ReAct yourself, you can debug ANY agent. You'll
recognize:
  * "Why is my agent looping forever?" -> step 4 isn't triggering;
    the LLM keeps asking for tools when it should be answering.
  * "Why is my agent ignoring tool results?" -> the ToolMessage
    isn't being appended back into the message history correctly.
  * "Why does my agent hallucinate tool calls?" -> we forgot to
    bind the tools to the LLM, so it doesn't know they exist.

The new LangChain primitives in this file
-----------------------------------------
1. `@tool` decorator: turns a Python function into a LangChain tool
   (a callable with name + description + typed input schema). The
   docstring becomes the description the LLM sees - WRITE IT WELL.

2. `llm.bind_tools([t1, t2, ...])`: returns a new LLM object that
   knows about these tools. When called, the LLM can emit
   "tool_calls" in addition to (or instead of) text content.

3. `tool_calls` on AI messages: when the LLM decides to use a tool,
   the response message has a non-empty `.tool_calls` list. Each
   item has {name, args, id}. We dispatch on this.

4. `ToolMessage`: the message type for a tool's RESULT, sent back
   to the LLM. Must include `tool_call_id` so the LLM knows which
   call it's a response to.

The new LangGraph primitives
----------------------------
5. `add_messages` reducer: a smarter version of `add` for chat
   message lists. Handles message IDs, deduping, in-place updates.
   You ALWAYS use this for a `messages` field in agent state.

6. `MessagesState`: a prebuilt TypedDict with just one field:
       messages: Annotated[list[AnyMessage], add_messages]
   We extend it when we need extra state fields.

Graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> A[agent<br/>llm_with_tools.invoke]
        A -.should_continue<br/>tool_calls present.-> T[tools<br/>execute each tool_call]
        T --> A
        A -.should_continue<br/>no tool_calls.-> E([END])
    ```

The 4-step ReAct loop (concept):

    ```mermaid
    flowchart TB
        L1["1. LLM reads conversation"] --> L2{"2. tool_calls<br/>or text?"}
        L2 -->|tool_calls| L3["3a. execute tool<br/>append ToolMessage<br/>loop back"]
        L3 --> L1
        L2 -->|text only| L4["3b. return final answer<br/>STOP"]
    ```

Message-history evolution across the loop:

    ```mermaid
    flowchart LR
        M0["msgs = [HumanMessage]"] --> A1[agent]
        A1 --> M1["msgs += AIMessage<br/>(with tool_calls)"]
        M1 --> T1[tools]
        T1 --> M2["msgs += ToolMessage<br/>(matching tool_call_id)"]
        M2 --> A2[agent]
        A2 --> M3["msgs += AIMessage<br/>(final answer, no tool_calls)"]
        M3 --> END([END])
    ```

This is a CYCLE (Stage 8). The termination condition is "LLM
returned a message with no tool_calls." Simple but powerful.
============================================================================
"""

import os
from typing import Literal
from datetime import datetime

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
# `tool` decorator + ToolMessage type
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
# MessagesState = prebuilt TypedDict with a `messages` field that uses
# the smart `add_messages` reducer. Almost every agent uses this.
from langgraph.graph.message import MessagesState


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"


# ---------------------------------------------------------------------------
# 1. DEFINE TOOLS
# ---------------------------------------------------------------------------
# A tool is just a Python function with the @tool decorator. The
# DOCSTRING is what the LLM reads when deciding whether to call it,
# so write it like a function spec for a colleague.

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information on any topic.

    Use this when the user asks about recent news, current events,
    specific facts, or anything that requires fresh online data.
    Returns 2-3 short snippets from the top results.

    Args:
        query: a focused web-search query (3-10 words).
    """
    tavily = TavilySearch(max_results=3, search_depth="basic")
    raw = tavily.invoke({"query": query})
    snippets = [
        f"- ({r.get('title','?')}) {r.get('content','')[:200]}"
        for r in raw.get("results", [])
    ]
    return "\n".join(snippets) or "(no results)"


@tool
def get_current_time() -> str:
    """Get the current date and time in ISO format.

    Use this when the user asks about today's date, the current
    time, or anything that requires knowing 'now'.
    """
    return datetime.now().isoformat()


# Collect tools into a list - we pass this list both to the LLM
# (so it knows what tools exist) and to our tool-executor node
# (so it knows how to dispatch). Keep them in sync.
TOOLS = [web_search, get_current_time]

# Build a name -> tool dict for fast dispatch in the tools node.
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 2. INSTANTIATE THE LLM AND BIND TOOLS
# ---------------------------------------------------------------------------
# `bind_tools` is the magic line. After this, when we call
# llm_with_tools.invoke(messages), the response can contain BOTH:
#   * .content  - text the LLM wants to say
#   * .tool_calls - list of {name, args, id} the LLM wants us to run
#
# Provider-side, this uses Gemini's native function-calling API.
# LangChain abstracts the provider-specific format.
base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = base_llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 3. THE AGENT NODE - asks the LLM what to do next
# ---------------------------------------------------------------------------
# The agent node is dead simple: pass all messages to the LLM, append
# its response to messages. The LLM decides whether to call a tool
# (response will have tool_calls) or answer (response will have only
# .content). The router (step 4) inspects the response to decide.
#
# Note we return {"messages": [response]} - a list with one message.
# Because the `messages` field uses the `add_messages` reducer,
# this APPENDS the new message to the existing history.
def agent_node(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])

    # Useful debug: did the LLM ask for a tool, or did it answer?
    if response.tool_calls:
        names = [tc["name"] for tc in response.tool_calls]
        print(f"[agent] LLM requested tool(s): {names}")
    else:
        print(f"[agent] LLM answered (no tool calls)")

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 4. THE ROUTER - "did the LLM ask for a tool, or are we done?"
# ---------------------------------------------------------------------------
# Pure function: read the LAST message; if it has tool_calls, route
# to the tools node; otherwise route to END.
#
# This is the standard ReAct termination check. Note we'd normally
# also add a guard like "if N > MAX_TOOL_CALLS: force END" to prevent
# runaway loops; we'll add that in production code.
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5. THE TOOLS NODE - executes whatever the LLM asked for
# ---------------------------------------------------------------------------
# This node:
#   1. Reads the most recent AI message (which must have tool_calls).
#   2. For each tool_call, looks up the tool, runs it, wraps the
#      result in a ToolMessage with the matching tool_call_id.
#   3. Returns those ToolMessages to be appended to messages.
#
# CRITICAL: the tool_call_id MUST match. The LLM uses it to correlate
# requests and responses. If you shuffle them or omit IDs, the model
# will be confused and may call tools again unnecessarily.
#
# In production you'd use `langgraph.prebuilt.ToolNode` instead of
# this hand-rolled version, but writing it yourself once removes
# all the mystery.
def tools_node(state: MessagesState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for call in last_message.tool_calls:
        name = call["name"]
        args = call["args"]
        call_id = call["id"]

        print(f"  [tools] running {name}({args})")
        tool = TOOLS_BY_NAME.get(name)

        if tool is None:
            # The LLM hallucinated a tool that doesn't exist. We tell
            # it so via a ToolMessage; the next agent turn can recover.
            content = f"Error: no tool named {name!r}"
        else:
            try:
                # Tools have a uniform .invoke(args_dict) interface.
                content = tool.invoke(args)
            except Exception as e:
                content = f"Error running {name}: {e}"

        tool_messages.append(
            ToolMessage(content=str(content), tool_call_id=call_id, name=name)
        )

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# 6. WIRE THE GRAPH
# ---------------------------------------------------------------------------
# The classic 2-node ReAct cycle:
#
#                         (no tool_calls)
#       START -> agent -> [should_continue] ────────> END
#                  ▲              │
#                  │              ▼ (tool_calls present)
#                  └────────── tools
#
# - The conditional edge from `agent` decides loop vs end.
# - The plain edge from `tools` always goes back to `agent`.
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")    # back to agent after tool runs

graph = builder.compile()


# ---------------------------------------------------------------------------
# 7. RUN
# ---------------------------------------------------------------------------
# Try a few queries:
#   * "What's the current time?"      -> 1 tool call, then answer
#   * "What is 2+2?"                  -> 0 tool calls, direct answer
#   * "Latest news on Gemini 3?"      -> 1 tool call, then answer
#   * "What's the time AND latest..." -> possibly 2 tool calls in one go
if __name__ == "__main__":
    questions = [
        "What is 2 + 2?",
        "What is the current date and time?",
        "What were the most-discussed AI papers on arxiv this past week?",
    ]

    for q in questions:
        print("\n" + "=" * 70)
        print(f"USER: {q}")
        print("=" * 70)

        # We just feed in a single user message. The agent loop takes
        # care of multi-turn tool calls internally.
        final_state = graph.invoke({"messages": [("user", q)]})

        # The final answer is the content of the LAST AI message
        # (the one without tool_calls - that's what made us exit).
        last = final_state["messages"][-1]
        print(f"\nASSISTANT: {last.content}")
        print(f"(total messages in conversation: {len(final_state['messages'])})")
