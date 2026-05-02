# Applied LangGraph — A Step-by-Step Curriculum

A hands-on, progressive tutorial for learning [LangGraph](https://langchain-ai.github.io/langgraph/) and modern agent engineering from scratch. Each script in `src/deep_research/` introduces one or two new concepts, building on the previous one, until we have a fully autonomous **Deep Research Agent Swarm** as the capstone.

> **Philosophy**
> Every file is a self-contained lesson. Comments explain *why*, not just *what*. Run each script, read the output, then move to the next. The capstone (the Deep Research Swarm) is the *destination* — but the goal is to deeply understand every primitive that goes into it, so you can build *any* agent system, not just this one.

---

## How this curriculum is structured

The path is split into **6 modules** of increasing complexity. You don't need to memorize the full plan upfront — just know that we always learn a concept in isolation on a tiny example *before* applying it to the swarm.

| Module | Theme | Stages | What you can build at the end |
|---|---|---|---|
| **I. LangGraph Fundamentals** | The graph runtime | 1–5 | A linear LLM + tool pipeline |
| **II. Graph Control Flow** | Parallelism, branching, cycles, subgraphs | 6–9 | A multi-step agent with loops |
| **III. Agent Patterns** | ReAct, streaming, human-in-the-loop, time travel | 10–13 | A real interactive tool-using agent |
| **IV. Memory & RAG** | Embeddings, Qdrant, RAG, Self-RAG, Agentic RAG, long-term memory | 14–19 ✅ | A research agent that learns over time |
| **V. Multi-Agent Systems** | Supervisor, network, handoffs, observability, cost control | 20–24 | Coordinated specialist agents |
| **VI. Capstone** | The Deep Research Agent Swarm | 25+ | The full system in [`PROJECT2_PLAN.md`](PROJECT2_PLAN.md) |

Each stage produces one runnable file. Each file imports and reuses ideas from earlier files, so you'll see the swarm taking shape gradually rather than all at once.

---

## 🗺️ Full Stage Map

Legend: ✅ done · 🚧 next · 📋 planned

### Module I — LangGraph Fundamentals

| Stage | File | Concepts | Status |
|---|---|---|---|
| 1 | [`hello_graph.py`](src/deep_research/module_1_fundamentals/01_hello_graph.py) | State (`TypedDict`), Nodes, Edges, `compile()`, `invoke()` | ✅ |
| 2–3 | [`two_node_graph.py`](src/deep_research/module_1_fundamentals/02_two_node_graph.py) | Reducers (`Annotated[list, add]`), multi-node sequencing | ✅ |
| 4–5 | [`llm_and_tools.py`](src/deep_research/module_1_fundamentals/03_llm_and_tools.py) | Real Gemini calls, structured output (Pydantic), Tavily tool calls | ✅ |

### Module II — Graph Control Flow

| Stage | File | Concepts | Status |
|---|---|---|---|
| 6 | [`parallel_fanout.py`](src/deep_research/module_2_control_flow/04_parallel_fanout.py) | `Send` API — fan out N searcher copies in parallel; reducers under load | ✅ |
| 7 | [`conditional_edges.py`](src/deep_research/module_2_control_flow/05_conditional_edges.py) | `add_conditional_edges()` for routing; branching logic; `END` as a route target | ✅ |
| 8 | [`cycles_and_command.py`](src/deep_research/module_2_control_flow/06_cycles_and_command.py) | Cycles in graphs; `Command(goto=..., update=...)` for handoff; the critic→searcher feedback loop | ✅ |
| 9 | [`subgraphs.py`](src/deep_research/module_2_control_flow/07_subgraphs.py) | Encapsulating an "agent" as a `StateGraph` you can drop into a parent graph; state translation between parent and child | ✅ |

### Module III — Agent Patterns

| Stage | File | Concepts | Status |
|---|---|---|---|
| 10 | [`streaming_modes.py`](src/deep_research/module_3_agent_patterns/08_streaming_modes.py) | `.stream()` vs `.invoke()`; the four stream modes (`values`, `updates`, `messages`, `custom`); building a live activity feed | ✅ |
| 11 | [`react_from_scratch.py`](src/deep_research/module_3_agent_patterns/09_react_from_scratch.py) | Build the ReAct loop yourself — `bind_tools`, `tool_calls`, `ToolMessage`, `MessagesState`, the cycle | ✅ |
| 12 | [`prebuilt_react_agent.py`](src/deep_research/module_3_agent_patterns/10_prebuilt_react_agent.py) | `create_react_agent` — system prompts, hooks, structured output, embedding as a subgraph; when to use vs hand-roll | ✅ |
| 13 | [`human_in_the_loop.py`](src/deep_research/module_3_agent_patterns/11_human_in_the_loop.py) | Checkpointing (`MemorySaver`), `interrupt()`, `Command(resume=...)`, time-travel via `get_state_history()` | ✅ |

### Module IV — Memory & RAG

The user explicitly asked for these. RAG belongs here because once an agent has memory + retrieval, the swarm becomes 10× more powerful.

| Stage | File | Concepts | Status |
|---|---|---|---|
| 14 | [`qdrant_basics.py`](src/deep_research/module_4_memory_and_rag/12_qdrant_basics.py) | Embeddings 101 (`gemini-embedding-001`), Qdrant collections, upsert + search, metadata filters — *no LangChain*, just the raw client | ✅ |
| 15 | [`naive_rag.py`](src/deep_research/module_4_memory_and_rag/13_naive_rag.py) | Classic 2-node RAG in LangGraph: retrieve → generate. Tagged-chunk prompt-injection defense. Citations by `source_id`. | ✅ |
| 16 | [`better_rag.py`](src/deep_research/module_4_memory_and_rag/14_better_rag.py) | Query rewriting (LLM expansion), hybrid search (dense + BM25 + RRF), LLM cross-encoder reranker | ✅ |
| 17 | [`self_rag.py`](src/deep_research/module_4_memory_and_rag/15_self_rag.py) | Self-RAG / CRAG: grade each retrieved doc, decide whether to re-search, regenerate query if needed — a *graph-shaped* RAG pipeline that beats one-shot RAG handily | ✅ |
| 18 | [`agentic_rag.py`](src/deep_research/module_4_memory_and_rag/16_agentic_rag.py) | The LLM itself decides *whether* to retrieve and *what* to retrieve, using the retriever as a tool. The bridge between RAG and full agents. | ✅ |
| 19 | [`long_term_memory.py`](src/deep_research/module_4_memory_and_rag/17_long_term_memory.py) | LangGraph's `BaseStore` (`InMemoryStore` → `PostgresStore`), `InjectedStore`, `pre_model_hook` for memory injection. Per-user namespaces. mem0 mapped onto the same shape. | ✅ |

### Module V — Multi-Agent Systems

| Stage | File | Concepts | Status |
|---|---|---|---|
| 20 | `supervisor_pattern.py` | One supervisor LLM routes to specialist agents (`langgraph-supervisor`); handoffs via `Command(goto=...)` | 📋 |
| 21 | `network_pattern.py` | Peer-to-peer agents that hand off to *each other* with no central boss; when to use this vs. supervisor | 📋 |
| 22 | `agent_handoffs.py` | Cross-graph handoffs with `Command(goto=..., graph=...)`; passing state slices between agents | 📋 |
| 23 | `observability_langfuse.py` | Wiring [Langfuse](https://langfuse.com) into LangGraph; tracing every LLM call, tool call, and node transition; cost ledgers | 📋 |
| 24 | `cost_and_concurrency.py` | `asyncio.Semaphore` to cap parallel calls; hard kill-switches in state (`max_cost_usd`, `max_invocations`); retry with exponential backoff | 📋 |

### Module VI — Capstone: Deep Research Agent Swarm

This is where everything we learned converges into the system from [`PROJECT2_PLAN.md`](PROJECT2_PLAN.md). Each stage below builds one tier of the architecture.

| Stage | File / Module | Concepts | Status |
|---|---|---|---|
| 25 | `swarm/state.py` | The full `ResearchState` — `Source`, `SubQuestion`, `VerifiedClaim`, citation graph, cost ledger | 📋 |
| 26 | `swarm/agents/planner.py` + `orchestrator.py` | Tier 1 coordination — hypothesis tree, sufficiency check, dispatching searchers via `Send` | 📋 |
| 27 | `swarm/agents/searcher.py` + `browser.py` | Tier 2 information gathering — restricted toolset, `<untrusted_content>` injection defense | 📋 |
| 28 | `swarm/agents/critic.py` + `fact_checker.py` | Tier 3 quality control — adversarial loop with `max_critic_rounds=2`, parallel claim verification | 📋 |
| 29 | `swarm/trust/scorer.py` | 5-dimensional weighted trust score (count, authority, agreement, recency, verdict) | 📋 |
| 30 | `swarm/agents/synthesizer.py` + `citation_formatter.py` | Tier 4 output — embedding-similarity citation check at ≥0.7 cosine | 📋 |
| 31 | `swarm/memory/` | `MemoryBackend` Protocol; mem0 self-hosted backend; `subq_cache`, `facts`, `prefs` namespaces | 📋 |
| 32 | `swarm/graph.py` | `build_graph()` wires every agent together with checkpointing, observability, kill-switches | 📋 |
| 33 | `eval/run_eval.py` | 50-question eval harness; single-agent baseline vs swarm; accuracy, citation precision, cost, latency | 📋 |

---

## 📖 Stage Notes (so far)

### Stage 1 — Hello Graph ([`hello_graph.py`](src/deep_research/module_1_fundamentals/01_hello_graph.py))

The smallest possible LangGraph program. Introduces the four building blocks every LangGraph app needs:

1. **State** — a `TypedDict` acting as a shared "whiteboard" that all nodes read from and write to.
2. **Node** — a plain Python function that receives the current state and returns a partial update dict.
3. **Edge** — a rule saying "go to this node next" (`START → echo → END`).
4. **Compile** — turns the blueprint into a runnable graph.

```
START ──▶ echo ──▶ END
```

**Key takeaway:** LangGraph nodes never mutate state directly — they return a dict of fields to merge. LangGraph handles the merging.

---

### Stage 2 & 3 — Reducers + Multi-Node Graphs ([`two_node_graph.py`](src/deep_research/module_1_fundamentals/02_two_node_graph.py))

Two critical concepts in one script.

**Reducers** — what happens when multiple nodes write to the same list field? Without a reducer the second write *overwrites* the first. By annotating with `Annotated[list[str], add]`, LangGraph uses Python's `operator.add` to *concatenate* instead. This is the single most important pattern for parallel agents.

**Sequential nodes** — wires a `planner` node (decomposes a query into sub-questions) into a `summarizer` node (reads those sub-questions and produces findings). Demonstrates that downstream nodes can read what upstream nodes wrote.

```
START ──▶ planner ──▶ summarizer ──▶ END
```

**Key takeaway:** every list field that multiple nodes touch needs a reducer, or data will be silently lost.

---

### Stage 4 & 5 — Real LLM + Real Tools ([`llm_and_tools.py`](src/deep_research/module_1_fundamentals/03_llm_and_tools.py))

Same graph shape as Stage 2/3, but every node now does real work.

**Stage 4 — Gemini via LangChain:**
- Load API keys from `.env` using `python-dotenv`
- Instantiate `ChatGoogleGenerativeAI`
- Use **structured output** via Pydantic: define a `SubQuestions` schema, call `llm.with_structured_output(SubQuestions)`, and get a typed Python object back — no regex parsing, no string splitting

**Stage 5 — Tavily web search:**
- Instantiate `TavilySearch` as a LangChain tool
- Call it directly from a node (`tavily.invoke({"query": ...})`)
- Compress raw search results into compact finding strings for the next node

```
START ──▶ planner (Gemini, structured output)
              ──▶ searcher (Tavily)
                      ──▶ summarizer (Gemini, free-form) ──▶ END
```

**Key takeaway:** `llm.with_structured_output(PydanticModel)` is the single most useful LangChain feature for agents — it turns free-form LLM text into typed, reliable Python objects.

---

### Stage 6 — Parallel fan-out with `Send` ([`parallel_fanout.py`](src/deep_research/module_2_control_flow/04_parallel_fanout.py))

The first file that goes beyond a linear pipeline. The planner produces N sub-questions, and we dispatch N searcher copies that run **at the same time** — the visible signal of concurrency is print order: every `[searcher] START` arrives before any `DONE`, and the `DONE`s arrive in non-deterministic order depending on which Tavily call finishes first.

**The new primitive — `Send`:**
A `Send(node_name, payload)` is a "go run THIS node with THIS specific input" instruction. Returning a *list* of `Send`s from a routing function tells LangGraph: "fan out — run all of these in parallel, then wait for all to finish, then continue."

```python
from langgraph.types import Send

def dispatch_searchers(state: ResearchState) -> list[Send]:
    return [
        Send("searcher", {"sub_question": sq})  # custom payload per copy
        for sq in state["sub_questions"]
    ]

builder.add_conditional_edges("planner", dispatch_searchers, ["searcher"])
```

**Why each Send carries its own payload (not the full state):** this is **context isolation** — each searcher only sees its own sub-question, not the rest of the world. Anthropic's research showed this is *literally* why multi-agent systems beat single agents: less context pollution per worker.

**Where reducers finally earn their keep:** Stage 2 introduced `Annotated[list, add]` but the reducer never had real work to do (single writer). Now N parallel searchers all return `{"findings": [...]}`. Without the reducer, the last one to finish overwrites everyone else's data. With `add`, every searcher's findings get appended.

```
                       ┌──> searcher (sub-q 1) ──┐
                       │                          │
   START -> planner ───┼──> searcher (sub-q 2) ──┼──> summarizer -> END
                       │                          │
                       └──> searcher (sub-q N) ──┘
```

**Key takeaway:** `Send` + reducers are the two halves of safe parallelism in LangGraph. The Send fans out; the reducer merges back. Forget either and you'll lose data silently.

---

### Stage 7 — Conditional edges & routing ([`conditional_edges.py`](src/deep_research/module_2_control_flow/05_conditional_edges.py))

Same `add_conditional_edges` API as Stage 6, but used for the more common case: **branching** — picking ONE of several next nodes based on state.

A tiny smart router classifies a query as `simple`, `needs_search`, or `harmful` and routes accordingly:

```python
def route_after_classify(state) -> Literal["simple_answer", "web_search", "refuse"]:
    if state["route"] == "simple":       return "simple_answer"
    if state["route"] == "needs_search": return "web_search"
    return "refuse"  # safe default for harmful / unexpected

builder.add_conditional_edges(
    "classify",
    route_after_classify,
    ["simple_answer", "web_search", "refuse"],  # all possible targets
)
```

```
                              ┌──> simple_answer ──┐
                              │                     │
   START -> classify_query ───┼──> web_search ──────┼──> END
                              │                     │
                              └──> refuse ──────────┘
```

**The unified mental model for `add_conditional_edges`:**

| Routing fn returns | Behavior |
|---|---|
| A string | Branch — pick that one next node |
| A list of strings | Run all of them (rare) |
| A list of `Send` | Fan out (Stage 6) |

**Two design rules for routers:**
1. **Routers are pure** — no side effects, no LLM calls. Compute decisions in a node, store them in state, then have the router *read* state.
2. **Always have a safe default** — if state contains an unexpected value, the router shouldn't crash.

**Key takeaway:** the same primitive (`add_conditional_edges`) handles all three flavors of dynamic flow — branching, multi-target, fan-out. Once you've internalized this, every LangGraph topology is just edges + routers.

---

### Stage 8 — Cycles & `Command` ([`cycles_and_command.py`](src/deep_research/module_2_control_flow/06_cycles_and_command.py))

Until now every graph was a DAG (data only flows forward). This stage breaks that and introduces the most powerful agent pattern: **self-correction via cycles.**

The critic reads the searchers' findings, spots gaps, and routes back UPSTREAM to the orchestrator with new follow-up questions. Capped at 2 rounds so it can't spin forever — that cap comes directly from PROJECT2_PLAN.md.

**The new primitive — `Command`:**
Until now, a node returned a state-update dict, and a *separate* routing function decided where to go next. `Command` lets a single node return BOTH:

```python
from langgraph.types import Command

def critic_node(state) -> Command[Literal["orchestrator", "synthesize"]]:
    verdict = critic_llm.invoke(...)

    if verdict.sufficient:
        return Command(goto="synthesize")

    if state["critic_round"] + 1 >= MAX_CRITIC_ROUNDS:
        return Command(goto="synthesize")  # forced exit

    return Command(
        goto="orchestrator",                       # CYCLE
        update={
            "pending_questions": verdict.follow_up_questions,
            "critic_round": state["critic_round"] + 1,
        },
    )
```

**Two things to notice:**
- `goto=` is the routing decision; `update=` is the state update
- When you use `Command`, you don't call `add_conditional_edges` for that node — the Command **is** the edge

```
           ┌───────────────────────┐
           │                       │   (loop, +1 round)
   START -> orchestrator -> searcher -> critic
                                          │
                                          └─> synthesize -> END
```

**The two termination patterns** (you need at least one or you have an infinite loop):
1. **Counter in state** — `if critic_round >= MAX_ROUNDS: stop` (we use this)
2. **LLM signal** — `if verdict.sufficient: stop`

We use both, in priority order: respect the LLM's "I'm satisfied" signal first, fall back to the counter.

**`Command` vs `add_conditional_edges` — when to use which:**

| Use `Command` | Use `add_conditional_edges` |
|---|---|
| Node decides routing AND updates state | Pure routing decision based on state |
| Cycles / handoffs / agent-to-agent jumps | Branching where state is already set |
| You want one place that owns "this node's contract" | You want routing logic isolated from work |

**Key takeaway:** cycles give agents the ability to course-correct. The pattern is always: detect a gap → return upstream with new info → cap the loop with a counter. This same pattern shows up in Self-RAG (Module IV), supervisor agents (Module V), and the capstone's critic loop.

---

### Stage 9 — Subgraphs ([`subgraphs.py`](src/deep_research/module_2_control_flow/07_subgraphs.py))

By Stage 8 our graphs already have 5+ nodes and are getting cluttered. The capstone has 7 agent types. If we wrote it all in one graph we'd drown. **Subgraphs** let us encapsulate "an agent" as a self-contained `StateGraph` and use it as a single node in a parent graph.

Each subgraph is independently testable, swappable, and reusable. This file builds a 2-layer architecture:

```
Outer graph:
  planner -> [searcher_agent] x N (parallel) -> synthesize

The searcher_agent is itself a 3-node subgraph:
  search -> filter -> compress
```

**Two ways to use a subgraph:**

**Way 1 — Same state schema as parent:** if the subgraph's state overlaps with the parent's, drop the compiled subgraph in directly:
```python
parent_builder.add_node("searcher_agent", compiled_searcher_subgraph)
```

**Way 2 — Different state schema (input/output transformation):** the more useful pattern for real systems. The subgraph has its own focused state; you wrap the call:

```python
def searcher_agent_node(state: SearcherSpawnInput) -> dict:
    # 1. Translate input
    inner_input = {"sub_question": state["sub_question"], ...}

    # 2. Run subgraph (just like any compiled graph)
    inner_final = searcher_agent.invoke(inner_input)

    # 3. Translate output back into parent state shape
    return {"summaries": [inner_final["summary"]]}
```

**Why Way 2 is the right default:** keeping each subgraph's state minimal is the *same idea* as Send payloads in Stage 6 — context isolation. Each agent only sees what it needs.

**Bonus property:** because a compiled subgraph is just a Pregel object, you can `.invoke()` it standalone for unit-testing — no need to run the whole parent graph. The Stage 9 file demonstrates this at the bottom.

**Key takeaway:** subgraphs are how we go from "one big graph" to "a system of agents." Every Tier 1–4 agent in the capstone (Planner, Searcher, Critic, Fact-Checker, Synthesizer) will be its own subgraph wired together by `swarm/graph.py`.

---

### Stage 10 — Streaming modes ([`streaming_modes.py`](src/deep_research/module_3_agent_patterns/08_streaming_modes.py))

So far we've only used `graph.invoke(state)` — it blocks until the whole graph finishes. That's fine for a CLI but useless for a UI: with parallel searchers running 30+ seconds, the user stares at a frozen screen. `graph.stream(state)` yields events **as they happen** so we can render an activity feed, stream LLM tokens, and react to internal events in real time.

**The four streaming modes** — they're not alternatives, you usually combine them:

| Mode | What it yields | Best for |
|---|---|---|
| `values` | The full state after every node | Debugging, replays |
| `updates` | The delta (what THAT node returned) after every node | Activity feeds, "step N completed" UI |
| `messages` | LLM tokens as they arrive (`AIMessageChunk`) | Streaming chat-style responses |
| `custom` | Arbitrary events nodes emit via `get_stream_writer()` | "tool_call_started", "cache_hit", "trust_score_87" |

**Custom events from inside a node:**
```python
from langgraph.config import get_stream_writer

def searcher_node(state):
    writer = get_stream_writer()
    writer({"event": "tavily_started", "sub_question": state["sub_question"]})
    raw = tavily.invoke(...)
    writer({"event": "tavily_finished", "n_results": len(raw["results"])})
    return {"findings": [...]}
```

**Combine multiple modes** by passing a list — each yielded item becomes a `(mode, data)` tuple:
```python
for mode, data in graph.stream(initial, stream_mode=["updates", "custom", "messages"]):
    if mode == "messages":
        chunk, meta = data
        if meta.get("langgraph_node") == "summarizer" and chunk.content:
            print(chunk.content, end="", flush=True)
```

**One gotcha:** `with_structured_output()` calls are NOT token-streamable — Gemini emits the whole JSON atomically. Free-form `.invoke()` calls ARE streamable. So the planner won't stream tokens, but the synthesizer will.

**Key takeaway:** the activity stream from `PROJECT2_PLAN.md` (`{"type": "agent_started", ...}`, `{"type": "tool_call", ...}`, etc.) is just `stream_mode="custom"` with structured event dicts. The whole live-UI capability is a few lines per node plus one consumer loop.

---

### Stage 11 — ReAct from scratch ([`react_from_scratch.py`](src/deep_research/module_3_agent_patterns/09_react_from_scratch.py))

ReAct = "Reasoning + Acting." It's the canonical pattern behind every tool-using agent (ChatGPT browsing, Claude with tools, our future Searcher). The loop is dead simple:

1. LLM reads the conversation
2. LLM either emits a **tool call** OR a **final text answer**
3. If tool call: run the tool, append the result to the conversation, loop to step 1
4. If text answer: stop and return it

We build this as a 2-node LangGraph cycle before using the prebuilt version (Stage 12), so the prebuilt one stops feeling like magic.

```
                       (no tool_calls)
   START -> agent -> [should_continue] ────────> END
              ▲              │
              │              ▼ (tool_calls present)
              └────────── tools
```

**The new LangChain primitives:**

| Primitive | Purpose |
|---|---|
| `@tool` decorator | Turns a Python function into a LangChain tool. The **docstring** becomes the tool's description — write it like a function spec for a colleague. |
| `llm.bind_tools([tools])` | Returns a new LLM that knows about the tools. Responses can now include `tool_calls`. |
| `response.tool_calls` | List of `{name, args, id}` — what the LLM wants you to run. Empty means "I'm done, here's the answer." |
| `ToolMessage(content, tool_call_id)` | The message type for a tool's RESULT. The `tool_call_id` MUST match the original request — that's how the LLM correlates them. |

**The new LangGraph primitives:**

| Primitive | Purpose |
|---|---|
| `add_messages` reducer | Smarter version of `add` for message lists — handles message IDs, deduping, in-place updates. ALWAYS use for a `messages` field. |
| `MessagesState` | Prebuilt `TypedDict` with one field: `messages: Annotated[list[AnyMessage], add_messages]`. Extend it when you need extra state. |

**The 5 functions you need to write a ReAct agent:**

```python
# 1. Tools (with good docstrings — the LLM reads them)
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information..."""

# 2. LLM with tools bound
llm_with_tools = base_llm.bind_tools([web_search, get_current_time])

# 3. Agent node — ask the LLM what to do
def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. Router — tool calls? or done?
def should_continue(state) -> Literal["tools", "__end__"]:
    return "tools" if state["messages"][-1].tool_calls else END

# 5. Tools node — execute whatever the LLM asked for
def tools_node(state):
    last = state["messages"][-1]
    return {"messages": [
        ToolMessage(content=str(TOOLS_BY_NAME[c["name"]].invoke(c["args"])),
                    tool_call_id=c["id"], name=c["name"])
        for c in last.tool_calls
    ]}
```

That's literally the whole agent. Five small functions and a loop.

**Three failure modes you'll instantly recognize after building this:**
- *"Why is my agent looping forever?"* → the `should_continue` check isn't firing — the LLM keeps calling tools when it should answer (often: missing system prompt instructing it to stop).
- *"Why is my agent ignoring tool results?"* → the `ToolMessage` isn't being appended back into messages, or `tool_call_id` doesn't match.
- *"Why does my agent hallucinate tool calls?"* → forgot to `bind_tools` to the LLM, so it doesn't actually know they exist.

**Key takeaway:** every "agent" is a ReAct loop. Master these 5 functions and you can debug or extend any agent system — including the Searcher, Browser, and Fact-Checker in our capstone.

---

### Stage 12 — The prebuilt ReAct agent ([`prebuilt_react_agent.py`](src/deep_research/module_3_agent_patterns/10_prebuilt_react_agent.py))

In Stage 11 we wrote ~80 lines for a basic agent. LangGraph ships a one-liner:

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model=llm, tools=TOOLS)
```

Same graph shape as our hand-rolled version, plus a production-grade `ToolNode` that parallelizes tool calls, catches exceptions, and handles streaming. We just learned the internals so the prebuilt stops feeling like magic — now we use it where it fits.

**What `create_react_agent` gives you for free:**

| Feature | Param |
|---|---|
| System prompt set once at creation | `prompt="..."` |
| Pre/post-LLM-call hooks (counting tokens, kill-switches, logging) | `pre_model_hook=`, `post_model_hook=` |
| Structured final output (Pydantic) | `response_format=PydanticModel` |
| Checkpointer + memory store passthrough | `checkpointer=`, `store=` |
| Drop-in as a subgraph in any parent graph | (it's just a Pregel) |

**Tutorial breakdown in the file:**

- **A — Minimal:** one line, run it, done.
- **B — Prompt + hook:** post-call hook logs every LLM invocation. This is exactly where the `max_tool_calls_per_agent = 8` kill-switch from `PROJECT2_PLAN.md` lives.
- **C — Structured output:** `response_format=SearchVerdict` makes the agent emit `{answer, confidence, sources}` as a Pydantic instance available at `state["structured_response"]`.
- **D — Agent as subgraph:** the capstone pattern. The compiled prebuilt agent is a Pregel object — drop it into a parent `StateGraph` like any other node, with input/output translation per Stage 9.

**The decision flowchart you should internalize:**

| Need | Use |
|---|---|
| Custom 3+ node loop (e.g. agent → validator → tools → agent) | Hand-roll (Stage 11) |
| `Send`-based parallel fan-out *inside* the agent | Hand-roll |
| State-shape transforms between LLM calls | Hand-roll |
| Otherwise | `create_react_agent` |

**Capstone agent split, roughly half-and-half:**

| Agent | Approach | Why |
|---|---|---|
| Searcher, Browser, Fact-Checker | Prebuilt | Simple loop, one tool, structured output |
| Critic | Hand-roll | Uses `Command` cycles (Stage 8) |
| Lead Orchestrator | Hand-roll | `Send` fan-out + sufficiency check |
| Synthesizer | Hand-roll | No tools, just structured output over big context |

**Key takeaway:** the prebuilt isn't a "use it for everything" tool — it's a "use it where it fits" tool. Stage 11 taught you what's in the box; Stage 12 teaches you when the box is the right shape.

---

### Stage 13 — Human-in-the-loop, checkpoints & time travel ([`human_in_the_loop.py`](src/deep_research/module_3_agent_patterns/11_human_in_the_loop.py))

Three powerful capabilities all powered by ONE underlying mechanism: **checkpointing**. After this stage you'll see why every production LangGraph app passes a checkpointer at compile time.

**The four primitives:**

| Primitive | What it does |
|---|---|
| `MemorySaver()` (and friends) | Saves the full graph state after every step, indexed by `thread_id`. `MemorySaver` is dev-only; production swaps in `SqliteSaver` / `AsyncPostgresSaver` with the same API. |
| `interrupt(payload)` inside a node | PAUSES the graph. Caller sees the payload (e.g. "approve this $5 spend?"). |
| `Command(resume=value)` | Resumes the paused graph. `value` becomes the return value of `interrupt()` inside the node. |
| `graph.get_state_history(config)` + `graph.update_state(...)` | Lists every checkpoint and lets you replay/edit from any past point. |

**The interrupt pattern:**
```python
def request_approval(state):
    decision = interrupt({                        # graph PAUSES here
        "search_plan": state["proposed_search"],
        "estimated_cost_usd": state["estimated_cost_usd"],
        "options": ["ok", "cheaper", "cancel"],
    })
    return {"approval": decision}                 # `decision` is whatever
                                                  # was passed in resume
```

**The two-call resume pattern:**
```python
config = {"configurable": {"thread_id": "user-123"}}

# 1) First invoke runs until interrupt
first = graph.invoke(initial_state, config=config)
print(first["__interrupt__"][0].value)            # show payload to human

# 2) Second invoke with Command(resume=...) picks up where it left off
final = graph.invoke(Command(resume="ok"), config=config)
```

**Time travel — replay or branch from any past checkpoint:**
```python
history = list(graph.get_state_history(config))
target = history[5]                               # some past checkpoint
# Resume from target.config with a *different* decision -> creates a new branch
final = graph.invoke(Command(resume="cheaper"), config=target.config)
```

**Why this is non-negotiable for the capstone:**

- **Resume-after-crash** — the swarm makes 20+ LLM calls. If call 19 fails on a transient API error, you do NOT want to redo calls 1–18.
- **Cost approvals** — *"about to spend $4.20 on this query, OK?"* is exactly an `interrupt()` use case.
- **Eval replays** — feed the same `thread_id` + initial state, get bit-for-bit identical traces.
- **The "rewind & rewrite" UI** from `PROJECT2_PLAN.md` section 10 is `get_state_history` + `update_state`.

**Key takeaway:** a checkpointer is the single most important production line:
```python
graph = builder.compile(checkpointer=MemorySaver())
```
Without it, no interrupts, no replays, no resumes. With it, you get all of agent ergonomics, debugging, and reliability for one extra argument.

---

### Stage 14 — Qdrant + embeddings, raw ([`qdrant_basics.py`](src/deep_research/module_4_memory_and_rag/12_qdrant_basics.py))

Before we let LangChain wrap it all in helpers, we use **`qdrant-client` and `google.genai` directly** — no LangChain in this file. The point is to demystify what RAG is actually doing underneath: take text, turn it into a vector, store it, search by meaning.

**The two-step mental model that all of RAG reduces to:**

```
"What's the weather like?"
        ↓ (1) embed
[0.013, -0.241, 0.882, ..., 0.005]  (768 numbers)
        ↓ (2) cosine similarity in Qdrant
[matching docs, sorted by relevance]
```

**The five primitives in this file** — every RAG paper/framework is just composing these:

| Primitive | What it does |
|---|---|
| `embed_text(text, task_type=...)` | Text → 768-D vector via `gemini-embedding-001` |
| `qdrant.create_collection(...)` | Define a "table" with vector size + distance metric (cosine) |
| `qdrant.upsert(points)` | Store `{id, vector, payload}` triples |
| `qdrant.query_points(query=qvec, ...)` | Top-K cosine search |
| `Filter(must=[FieldCondition(...)])` | Metadata constraints (`topic="qdrant"`, `year >= 2025`, etc.) |

**The asymmetric retrieval gotcha:** Gemini's embedding model produces *different* vectors for the same text depending on `task_type`:

```python
embed_text(doc,   task_type="RETRIEVAL_DOCUMENT")   # for stored text
embed_text(query, task_type="RETRIEVAL_QUERY")      # for user questions
```

Using these correctly gives meaningfully better retrieval. Mixing them up still works but ranks suboptimally — so just always use the right one.

**Why Qdrant specifically:** open-source, Rust-native (very fast at billion+ vectors), excellent payload filtering, and `PROJECT2_PLAN.md`'s mem0 config picks Qdrant as the vector store. We run it locally on `:6333` (already set up via Docker).

**Key takeaway:** RAG, Self-RAG, and Agentic RAG are *all* just clever orchestrations of the five operations in this file. Once these click, every framework's magic dissolves into "embed → upsert → search → filter." Stage 15 wires these into LangGraph; Stages 17–18 add intelligence around them.

---

### Stage 15 — Naive RAG in LangGraph ([`naive_rag.py`](src/deep_research/module_4_memory_and_rag/13_naive_rag.py))

Now we wire Stage 14's primitives into a LangGraph pipeline. **Naive RAG is the floor** — the simplest version that works, the baseline you'll see in 90% of "chatbot for your docs" tutorials.

```
START → retrieve → generate → END
```

**The two nodes:**

```python
def retrieve_node(state):
    qvec = embed_one(state["query"], task_type="RETRIEVAL_QUERY")
    res = qdrant.query_points(collection_name=COLLECTION, query=qvec, limit=4)
    return {"retrieved_chunks": [{"source_id": h.payload["source_id"],
                                  "text": h.payload["text"],
                                  "score": float(h.score)} for h in res.points]}

def generate_node(state):
    context = "\n\n".join(
        f'<retrieved_chunk source_id="{c["source_id"]}">\n{c["text"]}\n</retrieved_chunk>'
        for c in state["retrieved_chunks"]
    )
    prompt = "Answer using ONLY the chunks. Cite source_ids like [lg-02]. ..."
    return {"answer": llm.invoke(prompt + ...).content}
```

**Two production-grade habits this file establishes:**

1. **Ingestion is NOT in the query graph.** It's a separate setup function (`ensure_corpus_ingested`). Putting ingestion inline would re-embed the corpus on every query — wasteful and wrong.
2. **Wrap retrieved chunks in tagged blocks** with a system instruction that the contents are *data, not commands*:
   ```
   <retrieved_chunk source_id="lg-02">
   ...text from a stored doc...
   </retrieved_chunk>
   ```
   This is the same `<untrusted_content>` pattern from `PROJECT2_PLAN.md` section 5. Cheap, effective, and the right default any time you put external text into a prompt.

**The four limitations of naive RAG** — each fixed in a later stage:

| Limitation | Fixed in |
|---|---|
| Vague queries get bad retrievals | Stage 16 (query rewriting) |
| Top-K cosine ranks "sounds similar" over "is correct" | Stage 16 (reranking) |
| No way to detect "the docs don't actually answer this" | Stage 17 (Self-RAG, graded retrieval) |
| Retriever runs on every query even when not needed | Stage 18 (Agentic RAG) |

**Key takeaway:** RAG is genuinely just two nodes. The "intelligence" lives in the prompt, the corpus, and (later) the orchestration around retrieval. Master this baseline and the next three stages become incremental upgrades on a system you already understand.

---

### Stage 16 — Better RAG: rewriting, hybrid search, reranking ([`14_better_rag.py`](src/deep_research/module_4_memory_and_rag/14_better_rag.py))

Three orthogonal upgrades on top of naive RAG, each fixing one of Stage 15's failure modes. None touch the corpus or the embeddings — they all sit either before or after retrieval. You can comment any of them out and see how the answers degrade.

```
START → rewrite → hybrid_retrieve → rerank → generate → END
```

**1. Query rewriting (pre-retrieval).** A small LLM call expands one vague user question into 2–3 sharper standalone queries — different vocabulary, different angles. The original is always kept first as the safety net.

```python
class RewrittenQueries(BaseModel):
    queries: list[str]  # standalone search prompts, not paraphrases

def rewrite_query(q: str) -> list[str]:
    out = llm.with_structured_output(RewrittenQueries).invoke(REWRITE_PROMPT.format(q=q))
    return [q] + [r for r in out.queries if r.strip()]
```

**2. Hybrid search + RRF (during retrieval).** Per query, run BOTH dense (cosine on `gemini-embedding-001`) AND sparse (BM25 via `rank_bm25`) retrieval, then fuse all the ranked lists with **Reciprocal Rank Fusion**:

```
score(doc) = Σ over rankers of  1 / (60 + rank_in_that_ranker)
```

Why RRF is the right tool: it doesn't need calibrated scores from either ranker — just ranks. Cosine and BM25 are on completely different scales, but their *rankings* are comparable. RRF is robust, hyperparameter-free (`k=60` is the literature default), and trivial to extend with more rankers.

| Catches | Strength |
|---|---|
| Synonyms, paraphrases ("fan out" ≈ "parallel workers") | Dense embeddings |
| Exact terms, code identifiers ("Send API", "MemorySaver") | BM25 sparse |

**3. LLM cross-encoder reranking (post-retrieval).** Embeddings are a *bi-encoder* — they encode query and chunk separately and compare vectors. Fast (corpus is embedded once), but the query and chunk never see each other. A *cross-encoder* (or an LLM doing the same job) scores `(query, chunk)` jointly:

```python
class RerankItem(BaseModel):
    source_id: str
    score: int = Field(ge=0, le=10)
    why: str

def rerank_with_llm(query, candidates, top_n=4) -> list[dict]:
    out = llm.with_structured_output(RerankResult).invoke(
        RERANK_PROMPT.format(q=query, chunks=tagged_blocks(candidates))
    )
    # join scores back to candidates, sort desc, keep top_n
```

The standard pattern is "wide net, then narrow": fetch ~20 candidates with hybrid retrieval, rerank to keep the best 4. In production you'd swap the LLM for Cohere Rerank or BGE-reranker-v2 — same interface, smaller/faster model.

**Key takeaway:** the trio of upgrades attacks three different failure modes (vague query → rewriting; lexical mismatch → hybrid; topical-but-wrong order → rerank) and they compose cleanly. The capstone's mem0 lookups (`subq_cache`, `facts`) will use exactly this pipeline; the user's question is rewritten into multiple retrieval angles before hitting the cache.

---

### Stage 17 — Self-RAG: graded retrieval & re-search loop ([`15_self_rag.py`](src/deep_research/module_4_memory_and_rag/15_self_rag.py))

Stage 16 retrieves better, but it still **always answers** — even when the chunks don't actually contain the facts. Self-RAG adds a critic LLM that grades each retrieval on two axes (relevance + support) and either passes, loops back to re-retrieve with an aggressive rewrite, or **abstains** explicitly.

This is the **Stage 8 critic-loop pattern applied to the RAG domain.** Same `Command(goto=..., update=...)` machinery, same termination invariants, same `MAX_ROUNDS = 2` cap from `PROJECT2_PLAN.md`.

```
                    ┌──────────────── (loop, +1 round) ───────────────┐
                    │                                                  │
   START → rewrite → retrieve → grade  ─── sufficient ──→ generate → END
                              │
                              └─── insufficient + budget exhausted ──→ abstain → END
```

**The critic returns a structured verdict:**

```python
class GradeVerdict(BaseModel):
    relevance_avg:   float = Field(ge=0, le=10)   # avg how-on-topic
    supports_answer: bool                          # do chunks contain the facts?
    missing:         str                           # one-line note for next rewrite
```

**The decision logic** (lives in a `Command`-returning grade node — no `add_conditional_edges` needed):

```python
def grade_node(state) -> Command[Literal["rewrite", "generate", "abstain"]]:
    verdict = grade_retrieval(state["query"], state["top_chunks"])

    if verdict.supports_answer and verdict.relevance_avg >= MIN_RELEVANCE:
        return Command(goto="generate", update={"grade": ...})

    if state["round_idx"] + 1 > MAX_ROUNDS:
        return Command(goto="abstain", update={"grade": ...})

    return Command(
        goto="rewrite",
        update={"grade": ..., "round_idx": state["round_idx"] + 1},
    )
```

**Round-aware rewriting.** The first round uses Stage 16's normal rewrite prompt. On retry, an *aggressive* rewrite prompt is fed the previous queries and the critic's `missing` note, and is told to try different vocabulary, break the question into sub-questions, or guess the corpus's actual terms.

**Why `abstain` is its own node, not just text in `generate`:**

- It's a first-class outcome in eval traces (you can grep for it).
- It's the right place to fall back to a different strategy (web search, escalate to a more expensive model, queue for "needs more docs").
- It enforces the single most important property of a good RAG system: *answer or don't, but never confidently fabricate.*

**Termination invariants** (always check these for any cyclic graph):

- Every path from `grade` either terminates (`generate` / `abstain`) or *strictly increments* the round counter.
- `MAX_ROUNDS` is a hard ceiling — even if the critic insists, we stop. Belt-and-braces, same as `PROJECT2_PLAN.md`.
- The state carries `history` breadcrumbs so debugging a long loop is one print away.

**Key takeaway:** cycles in graphs are how you get *self-correcting* retrieval. The exact same pattern shows up in the capstone's Critic node grading sufficiency every N sub-questions — the only difference is the unit being graded (whole findings, not chunks) and the loop body (dispatching more Searchers, not rewriting one query). Master this in 80 lines here, scale it up there.

---

### Stage 18 — Agentic RAG: retriever-as-tool ([`16_agentic_rag.py`](src/deep_research/module_4_memory_and_rag/16_agentic_rag.py))

Stages 15–17 all assume retrieval runs **on every query**. Stage 18 flips the control flow: the retriever becomes a `@tool` and the LLM decides if/when/with-what to call it. This is the bridge between "RAG" and "full agent" — and structurally it's just Stage 12's prebuilt ReAct agent with two carefully-written tools.

```
START → agent ↔ tools (retrieve_kb / lookup_by_topic) → END
```

**The two tools:**

```python
@tool
def retrieve_kb(query: str, top_n: int = 4) -> str:
    """Search the internal knowledge base for chunks relevant to `query`.
    USE THIS WHEN: ...
    DO NOT USE FOR: ...
    Returns: tagged <retrieved_chunk source_id="..."> blocks with
    rerank_score visible. Cite by source_id like [lg-02].
    """
    chunks = _hybrid_search_and_rerank(query, top_n=top_n)
    return tagged_blocks(chunks)

@tool
def lookup_by_topic(topic: str) -> str:
    """List ALL chunks tagged with a specific topic, no semantic search.
    USE THIS WHEN: exhaustive coverage of a known topic.
    Available topics: langgraph, qdrant, rag, agents, embeddings."""
```

**Two design rules this file establishes:**

1. **Tool docstrings ARE prompts.** The LLM picks tools based on the docstring alone. Write them like a function spec for a careful colleague: when to use, when NOT to use, argument semantics, return shape.
2. **Tools return strings, not Python objects.** The LLM only sees `ToolMessage.content`. Format as tagged blocks with `source_id` and `rerank_score` visible so the model can cite AND self-assess "did I get a good hit?".

**The system prompt encodes three policies you'll always need in agentic-RAG:**

- **When to retrieve vs answer directly** — "for trivial / off-topic, ANSWER DIRECTLY; for KB-relevant, CALL `retrieve_kb`."
- **Citation discipline** — "after every factual sentence, append [source_id]; only cite source_ids that appeared in tool results."
- **Abstention** — "if every chunk has rerank_score < 6, say plainly the KB doesn't cover this. Do NOT invent."

**Behaviour you'll see at runtime:**

| Question | Tool calls |
|---|---|
| "What's 2 + 2?" | 0 |
| "How does LangGraph fan out parallel workers?" | 1 (`retrieve_kb`) |
| "Compare LangGraph parallelism vs cycles." | 2 (`retrieve_kb` per side) |
| "Boiling point of mercury?" | 0 or 1 (often retrieves, sees weak hits, abstains) |

**Key takeaway:** this file IS the capstone Searcher's shape. Add `mem0_read` and `tavily_search` to `TOOLS` and you have ~60% of `PROJECT2_PLAN.md` sec 5 already in your hand.

---

### Stage 19 — Long-term memory: `BaseStore` and mem0 ([`17_long_term_memory.py`](src/deep_research/module_4_memory_and_rag/17_long_term_memory.py))

Every agent so far has been amnesiac across sessions. Stage 13's `MemorySaver` was the exception — it persists ONE thread for resume / replay — but checkpointers are per-thread by design. They don't cover cross-session needs:

- *"remember Alaap prefers concise, technical answers"*
- *"we already retrieved this fact 5 minutes ago — reuse it"*
- *"this user has a project called 'capstone-swarm'"*

That's what **`BaseStore`** is for. The two-layer memory model every production agent uses:

| | Stage 13 — Checkpointer | Stage 19 — Store |
|---|---|---|
| Scope | one thread of one session | spans threads & sessions |
| Unit | full graph state | freeform key-value items |
| Indexed by | `thread_id` | namespace tuple |
| Enables | resume, replay, time travel | user prefs, learned facts, cache, mem0 |
| Backends | `MemorySaver` → `SqliteSaver` → `AsyncPostgresSaver` | `InMemoryStore` → `PostgresStore` → custom (mem0) |

**The three operations you'll use 99% of the time:**

```python
store.put(namespace, key, value)            # upsert one item
store.get(namespace, key)                   # exact-key lookup
store.search(namespace, query=..., limit=)  # semantic search
```

`namespace` is a **tuple** — typically `(category, user_id)` so each user has isolated memory. With `index={"embed": fn, "dims": ..., "fields": [...]}` at construction, `store.search(query=...)` becomes semantic; without it, only metadata filters.

**The three namespaces from `PROJECT2_PLAN.md` sec 3 D2:**

```
("prefs",      user_id) -> {citation_style, depth, ...}
("subq_cache", user_id) -> {q -> top_chunks, ts}
("facts",      user_id) -> {claim, sources, trust}
```

**Wiring memory into the agent — three new primitives:**

1. **`store=` at `create_react_agent`** — plumbs the store into every tool that asks for it.
2. **`InjectedStore` in tool args** — lets a tool receive the runtime store without exposing it to the LLM:
   ```python
   @tool
   def retrieve_kb_cached(
       query: str,
       config: RunnableConfig,
       store: Annotated[BaseStore, InjectedStore()],
   ) -> str:
       user_id = config["configurable"]["user_id"]
       hits = store.search(("subq_cache", user_id), query=query, limit=1)
       if hits and hits[0].score >= 0.85:
           return hits[0].value["chunks_blob"]   # CACHE HIT
       # ... fall through to hybrid+rerank, then store.put(...) the result
   ```
3. **`pre_model_hook`** — runs before every LLM call. We use it to fetch all `prefs` + the top-3 semantically-relevant `facts` for the user's latest message, then prepend them as a `[memory]` system message. The same hook will host the kill-switches in Module 5.

**Per-call context via `RunnableConfig`** — pass `config={"configurable": {"user_id": "alaap"}}` on `invoke`; tools read it to scope their reads/writes. Free multi-tenant.

**mem0 — what it adds on top of `BaseStore`:**

1. **LLM-driven write compression.** `mem.add(messages=...)` runs an internal LLM that extracts durable facts, dedupes against existing memories, and only updates what's new. mem0 claims ~80% prompt-token reduction for memory injections.
2. **Scoped keys baked into the API** — every call takes `user_id`, optional `agent_id`, optional `session_id`. Same idea as namespace tuples, just first-class.
3. **Optional graph memory** — Neo4j-backed entity/relation graph across stored facts. `PROJECT2_PLAN.md` keeps this off for v1.

The ergonomic match between `BaseStore` and mem0 is why the capstone hides both behind a `MemoryBackend` Protocol — the swap is a few lines:

```python
# BaseStore
store.put(("facts", "alaap"), key=..., value=...)
store.search(("facts", "alaap"), query="...")

# mem0
mem.add(messages=..., user_id="alaap", metadata={"category": "facts"})
mem.search("...", user_id="alaap", filters={"category": "facts"})
```

**Key takeaway:** every production agent uses BOTH layers — checkpointer for "this thread's working memory", store for "everything that should outlive this thread." Master `BaseStore`'s `put` / `get` / `search` + namespaces and you've understood 90% of what mem0 is doing under the hood. Module IV is now complete; Module V composes these RAG-aware memory-aware agents into multi-agent systems.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Conda (recommended)
- API keys: [Google Gemini](https://aistudio.google.com/apikey) and [Tavily](https://tavily.com/)
- For Module IV+: a local Qdrant on `:6333` and Postgres on `:5432` (the curriculum's `docker-compose.yml` will provide these when we get there)

### Setup

```bash
# Clone
git clone https://github.com/alaap001/Applied-LangGraph.git
cd Applied-LangGraph

# Environment
conda create -n env_py312 python=3.12 -y
conda activate env_py312

# Core deps for Modules I–II
pip install langgraph langchain-google-genai langchain-tavily python-dotenv pydantic
```

Additional dependencies will be installed module-by-module as we introduce them:
- Module III: `langchain` (for `create_react_agent`, `ToolNode`)
- Module IV: `qdrant-client`, `langchain-qdrant`, `rank_bm25`, `mem0ai`
- Module V: `langfuse`, `langgraph-supervisor`

### Configure API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"
LANGFUSE_SECRET_KEY="..."          # Module V onward
LANGFUSE_PUBLIC_KEY="..."          # Module V onward
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
```

### Run the tutorials

```bash
# Module I — Stages 1, 2/3, 4/5
python src/deep_research/module_1_fundamentals/01_hello_graph.py
python src/deep_research/module_1_fundamentals/02_two_node_graph.py
python src/deep_research/module_1_fundamentals/03_llm_and_tools.py

# Module II — Stages 6, 7, 8, 9
python src/deep_research/module_2_control_flow/04_parallel_fanout.py
python src/deep_research/module_2_control_flow/05_conditional_edges.py
python src/deep_research/module_2_control_flow/06_cycles_and_command.py
python src/deep_research/module_2_control_flow/07_subgraphs.py

# Module III — Stages 10, 11, 12, 13
python src/deep_research/module_3_agent_patterns/08_streaming_modes.py
python src/deep_research/module_3_agent_patterns/09_react_from_scratch.py
python src/deep_research/module_3_agent_patterns/10_prebuilt_react_agent.py
python src/deep_research/module_3_agent_patterns/11_human_in_the_loop.py

# Module IV — Stages 14, 15, 16, 17, 18, 19  (requires Qdrant on :6333)
python src/deep_research/module_4_memory_and_rag/12_qdrant_basics.py
python src/deep_research/module_4_memory_and_rag/13_naive_rag.py
python src/deep_research/module_4_memory_and_rag/14_better_rag.py
python src/deep_research/module_4_memory_and_rag/15_self_rag.py
python src/deep_research/module_4_memory_and_rag/16_agentic_rag.py
python src/deep_research/module_4_memory_and_rag/17_long_term_memory.py
```

---

## 🏗️ Where this is heading

The capstone is a **Deep Research Agent Swarm** — a hierarchical multi-agent system with:

- **7 specialized agent types** (Planner, Orchestrator, Searcher, Browser, Critic, Fact-Checker, Synthesizer)
- **Parallel fan-out** via LangGraph's `Send` API (5–15 Searchers running simultaneously)
- **Adversarial self-correction** via a Critic → Searcher feedback loop
- **Trust scoring** on every claim (source count, authority, agreement, recency, fact-checker verdict)
- **Dual-layer memory** — per-session checkpointing + cross-session semantic memory (mem0)
- **Embedding-verified citations** (≥0.7 cosine between claim and source)
- **Hard cost & rate-limit controls** so it can't run away

Each upcoming stage introduces the LangGraph concept(s) needed to build one slice of this system, keeping the same learn-by-doing approach. The full architecture is documented in [`PROJECT2_PLAN.md`](PROJECT2_PLAN.md).

---

## 🧠 Why this curriculum exists

Most LangGraph tutorials either show you a 5-line "hello agent" and stop, or dump a 500-line multi-agent system on you and call it a day. Neither teaches you how to *think* in graphs. This curriculum is the missing middle: every concept gets its own tiny file, and every file's only job is to make one new idea click. By the time we build the swarm, there's nothing in it that's still magic.

---

## 📂 Project Structure (current + planned)

```
applied-langgraph/
├── README.md                                # ← you are here
├── PROJECT2_PLAN.md                         # capstone architecture
├── .env                                     # API keys (git-ignored)
├── pyproject.toml                           # deps
├── docker-compose.yml                       # Qdrant + Postgres (Module IV+)
├── src/
│   └── deep_research/
│       ├── module_1_fundamentals/
│       │   ├── README.md                    # module overview + diagrams
│       │   ├── 01_hello_graph.py            ✅
│       │   ├── 02_two_node_graph.py         ✅
│       │   └── 03_llm_and_tools.py          ✅
│       ├── module_2_control_flow/
│       │   ├── README.md
│       │   ├── 04_parallel_fanout.py        ✅
│       │   ├── 05_conditional_edges.py      ✅
│       │   ├── 06_cycles_and_command.py     ✅
│       │   └── 07_subgraphs.py              ✅
│       ├── module_3_agent_patterns/
│       │   ├── README.md
│       │   ├── 08_streaming_modes.py        ✅
│       │   ├── 09_react_from_scratch.py     ✅
│       │   ├── 10_prebuilt_react_agent.py   ✅
│       │   └── 11_human_in_the_loop.py      ✅
│       ├── module_4_memory_and_rag/
│       │   ├── README.md
│       │   ├── 12_qdrant_basics.py          ✅
│       │   ├── 13_naive_rag.py              ✅
│       │   ├── 14_better_rag.py             ✅
│       │   ├── 15_self_rag.py               ✅
│       │   ├── 16_agentic_rag.py            ✅
│       │   └── 17_long_term_memory.py       ✅
│       ├── module_5_multi_agent/            📋
│       │   ├── 18_supervisor_pattern.py
│       │   ├── 19_network_pattern.py
│       │   ├── 20_agent_handoffs.py
│       │   ├── 21_observability_langfuse.py
│       │   └── 22_cost_and_concurrency.py
│       └── module_6_capstone/               📋
│           └── swarm/                       # Stages 25-32 (capstone)
│               ├── state.py
│               ├── memory/
│               ├── tools/
│               ├── agents/
│               ├── trust/
│               └── graph.py
├── eval/                                    # Stage 33
│   ├── questions.jsonl
│   └── run_eval.py
└── tests/
```

**Every module has its own `README.md`** with module-level diagrams and a links table; every `.py` file has Mermaid diagrams in its top docstring (graph topology + concept diagrams). GitHub renders Mermaid natively in both `.md` files and the docstring code-blocks.

---

## 📚 Reference docs we lean on

- [LangGraph docs](https://langchain-ai.github.io/langgraph/) — graph runtime, `Send`, `Command`, checkpointing
- [LangChain docs](https://python.langchain.com/) — chat models, tools, structured output
- [Qdrant docs](https://qdrant.tech/documentation/) — vector store fundamentals
- [mem0 docs](https://docs.mem0.ai/) — long-term agent memory
- [Anthropic's multi-agent research blog](https://www.anthropic.com/engineering/built-multi-agent-research-system) — the architectural inspiration for the capstone
