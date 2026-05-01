# Module 1 — LangGraph Fundamentals

The four building blocks of every LangGraph app, introduced one or two at a time.

| File | Stage | Concepts |
|---|---|---|
| [`01_hello_graph.py`](01_hello_graph.py) | 1 | State (`TypedDict`), Nodes, Edges, `compile()`, `invoke()` |
| [`02_two_node_graph.py`](02_two_node_graph.py) | 2–3 | Reducers (`Annotated[list, add]`), multi-node sequencing |
| [`03_llm_and_tools.py`](03_llm_and_tools.py) | 4–5 | Real Gemini calls, structured output (Pydantic), Tavily tool calls |

---

## The 4 building blocks

```mermaid
flowchart LR
    State["STATE<br/>TypedDict whiteboard<br/>all nodes read/write"]
    Nodes["NODES<br/>plain Python fns<br/>state -> partial update"]
    Edges["EDGES<br/>routing rules<br/>'after A, go to B'"]
    Compile["COMPILE<br/>blueprint -> runnable"]
    State --> Nodes --> Edges --> Compile --> Run["graph.invoke(state)"]
```

## Module-end graph (Stage 4–5)

```mermaid
flowchart LR
    S([START]) --> P[planner<br/>Gemini + structured output]
    P --> SE[searcher<br/>Tavily search]
    SE --> SU[summarizer<br/>Gemini free-form]
    SU --> E([END])
```

## Module 1 → Module 2 cliffhanger

By the end of Stage 5 we have a linear DAG. Module 2 introduces parallelism, branching, cycles, and subgraphs — the four control-flow patterns that turn a pipeline into a system.
