# Applied LangGraph — A Step-by-Step Walkthrough

A hands-on, progressive tutorial for learning [LangGraph](https://langchain-ai.github.io/langgraph/) from scratch. Each script in `src/deep_research/` introduces exactly one or two new concepts, building on the previous one, until we have a fully autonomous Deep Research Agent Swarm.

> **Philosophy:** Every file is designed to be read top-to-bottom like a lesson. Comments explain *why*, not just *what*. Run each script, read the output, then move to the next.

---

## 🗺️ Learning Path

| Stage | File | Concepts Introduced | Complexity |
|-------|------|---------------------|------------|
| 1 | [`hello_graph.py`](src/deep_research/hello_graph.py) | State, Nodes, Edges, Compile & Invoke | ⭐ |
| 2–3 | [`two_node_graph.py`](src/deep_research/two_node_graph.py) | Reducers (`Annotated[list, add]`), Multi-node sequencing | ⭐⭐ |
| 4–5 | [`llm_and_tools.py`](src/deep_research/llm_and_tools.py) | Real LLM (Gemini), Structured Output (Pydantic), Tool Calling (Tavily) | ⭐⭐⭐ |
| 6 | *Coming soon* | Parallel fan-out with `Send` API | ⭐⭐⭐ |
| 7 | *Coming soon* | Conditional edges & routing | ⭐⭐⭐⭐ |
| 8 | *Coming soon* | Cycles, Critic loop with `Command` | ⭐⭐⭐⭐ |
| 9 | *Coming soon* | Checkpointing & memory | ⭐⭐⭐⭐ |
| 10 | *Coming soon* | Full Deep Research Agent Swarm | ⭐⭐⭐⭐⭐ |

---

## 📖 Stage Breakdown

### Stage 1 — Hello Graph ([`hello_graph.py`](src/deep_research/hello_graph.py))

The smallest possible LangGraph program. Introduces the 4 building blocks every LangGraph app needs:

1. **State** — A `TypedDict` acting as a shared "whiteboard" that all nodes read from and write to.
2. **Node** — A plain Python function that receives the current state and returns a partial update dict.
3. **Edge** — A rule saying "go to this node next" (`START → echo → END`).
4. **Compile** — Turns the blueprint into a runnable graph.

```
START ──▶ echo ──▶ END
```

**Key takeaway:** LangGraph nodes never mutate state directly — they return a dict of fields to merge. LangGraph handles the merging.

---

### Stage 2 & 3 — Reducers + Multi-Node Graphs ([`two_node_graph.py`](src/deep_research/two_node_graph.py))

Two critical concepts combined in one script:

**Reducers** — What happens when multiple nodes write to the same list field? Without a reducer the second write *overwrites* the first. By annotating with `Annotated[list[str], add]`, LangGraph uses Python's `operator.add` to *concatenate* instead. This is the single most important pattern for parallel agents.

**Sequential nodes** — Wires a `planner` node (decomposes a query into sub-questions) into a `summarizer` node (reads those sub-questions and produces findings). Demonstrates that downstream nodes can read what upstream nodes wrote.

```
START ──▶ planner ──▶ summarizer ──▶ END
```

**Key takeaway:** Every list field that multiple nodes touch needs a reducer, or data will be silently lost.

---

### Stage 4 & 5 — Real LLM + Real Tools ([`llm_and_tools.py`](src/deep_research/llm_and_tools.py))

Same graph shape as Stage 2/3, but every node now does real work:

**Stage 4 — Gemini via LangChain:**
- Load API keys from `.env` using `python-dotenv`
- Instantiate `ChatGoogleGenerativeAI` (Gemini 3.1 Pro)
- Use **structured output** via Pydantic: define a `SubQuestions` schema, call `llm.with_structured_output(SubQuestions)`, and get a typed Python object back — no regex parsing, no string splitting

**Stage 5 — Tavily Web Search:**
- Instantiate `TavilySearch` as a LangChain tool
- Call it directly from a node (`tavily.invoke({"query": ...})`)
- Compress raw search results into compact finding strings for the next node

```
START ──▶ planner (Gemini structured output)
              ──▶ searcher (Tavily web search)
                      ──▶ summarizer (Gemini free-form) ──▶ END
```

**Key takeaway:** `llm.with_structured_output(PydanticModel)` is the single most useful LangChain feature for agents — it turns free-form LLM text into typed, reliable Python objects.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Conda (recommended)
- API Keys: [Google Gemini](https://aistudio.google.com/apikey) and [Tavily](https://tavily.com/)

### Setup

```bash
# Clone
git clone https://github.com/alaap001/Applied-LangGraph.git
cd Applied-LangGraph

# Create environment
conda create -n env_py312 python=3.12 -y
conda activate env_py312

# Install dependencies
pip install langgraph langchain-google-genai langchain-tavily python-dotenv pydantic
```

### Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### Run the Tutorials

Execute the stages in order:

```bash
# Stage 1 — no API keys needed
python src/deep_research/hello_graph.py

# Stage 2 & 3 — no API keys needed
python src/deep_research/two_node_graph.py

# Stage 4 & 5 — requires GOOGLE_API_KEY + TAVILY_API_KEY
python src/deep_research/llm_and_tools.py
```

---

## 🏗️ Where This Is Heading

The tutorials above are the foundation. The end goal is a **Deep Research Agent Swarm** — a hierarchical multi-agent system with:

- **7 specialized agent types** (Planner, Orchestrator, Searcher, Browser, Critic, Fact-Checker, Synthesizer)
- **Parallel fan-out** via LangGraph's `Send` API (5–15 Searchers running simultaneously)
- **Adversarial self-correction** via a Critic → Searcher feedback loop
- **Trust scoring** on every claim (source count, authority, agreement, recency, fact-checker verdict)
- **Dual-layer memory** — per-session checkpointing + cross-session semantic memory (mem0)

Each upcoming stage will introduce the LangGraph concepts needed to build one piece of this system, keeping the same learn-by-doing approach.

The full architecture is documented in [`PROJECT2_PLAN.md`](PROJECT2_PLAN.md).

---