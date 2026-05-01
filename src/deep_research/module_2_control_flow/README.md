# Module 2 — Graph Control Flow

The four control-flow patterns that take you from "a pipeline" to "a system": parallelism, branching, cycles, and encapsulation.

| File | Stage | Concepts |
|---|---|---|
| [`04_parallel_fanout.py`](04_parallel_fanout.py) | 6 | `Send` API — fan out N workers in parallel; reducers under load |
| [`05_conditional_edges.py`](05_conditional_edges.py) | 7 | `add_conditional_edges()` for routing/branching |
| [`06_cycles_and_command.py`](06_cycles_and_command.py) | 8 | Cycles + `Command(goto=..., update=...)` for handoffs |
| [`07_subgraphs.py`](07_subgraphs.py) | 9 | Encapsulating an "agent" as a `StateGraph`, embed in parent |

---

## The four patterns at a glance

```mermaid
flowchart TB
    subgraph Fan["Stage 6: Parallel fan-out (Send)"]
        F1[node A] -.->|"Send #1"| F2[copy 1]
        F1 -.->|"Send #2"| F3[copy 2]
        F1 -.->|"Send #N"| F4[copy N]
        F2 --> F5[join]
        F3 --> F5
        F4 --> F5
    end
    subgraph Branch["Stage 7: Conditional branching"]
        B1[node A] -->|"route='X'"| B2[node X]
        B1 -.->|"route='Y'"| B3[node Y]
    end
    subgraph Cycle["Stage 8: Cycles + Command"]
        C1[node A] --> C2[node B]
        C2 --> C3[critic]
        C3 -.Command goto.-> C1
        C3 -.->|done| C4[END]
    end
    subgraph Sub["Stage 9: Subgraphs"]
        SG["agent_subgraph<br/>(its own internal graph)"] -.embedded as.- N1[node]
        N1 --> SG2[parent next]
    end
```

## The unified API mental model

`add_conditional_edges` is one API with three uses, depending on what the routing function returns:

```mermaid
flowchart LR
    F["routing fn"] -->|returns string| BR[branch]
    F -->|returns list of strings| MULTI[multi-target]
    F -->|returns list of Sends| FAN[fan-out]
```

`Command` is a *separate* primitive that fuses state-update + routing into one return value — used inside nodes (commonly for cycles).

## Module-end shape (Stage 9 capstone preview)

```mermaid
flowchart LR
    S([START]) --> P[planner]
    P -.Send.-> A1[searcher_agent #1<br/>= subgraph]
    P -.-> A2[searcher_agent #2<br/>= subgraph]
    P -.-> A3[searcher_agent #N<br/>= subgraph]
    A1 --> SY[synthesize]
    A2 --> SY
    A3 --> SY
    SY --> E([END])
```

This shape is already 80% of what the capstone swarm looks like. From here on, each later stage adds quality-control machinery (Critic, Fact-Checker, trust scoring) on top of this skeleton.
