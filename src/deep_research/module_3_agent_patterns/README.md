# Module 3 — Agent Patterns

Now that we know graphs, we build agents that *think and act*: tool-using ReAct loops, real-time streaming, and human-in-the-loop checkpointing.

| File | Stage | Concepts |
|---|---|---|
| [`08_streaming_modes.py`](08_streaming_modes.py) | 10 | `.stream()` modes: `values`, `updates`, `messages`, `custom` |
| [`09_react_from_scratch.py`](09_react_from_scratch.py) | 11 | Build the ReAct loop yourself: `bind_tools`, `tool_calls`, `ToolMessage` |
| [`10_prebuilt_react_agent.py`](10_prebuilt_react_agent.py) | 12 | `create_react_agent` — system prompt, hooks, structured output, embed as subgraph |
| [`11_human_in_the_loop.py`](11_human_in_the_loop.py) | 13 | Checkpointing, `interrupt()`, `Command(resume=...)`, time-travel |

---

## The ReAct loop (the spine of every tool-using agent)

```mermaid
flowchart LR
    S([START]) --> A[agent<br/>llm_with_tools.invoke]
    A -.tool_calls.-> T[tools<br/>execute, return ToolMessages]
    T --> A
    A -.no tool_calls.-> E([END])
```

## Streaming modes - which yields what

```mermaid
flowchart TB
    G["graph.stream(state, stream_mode=?)"] --> V["'values'<br/>full state per step"]
    G --> U["'updates'<br/>partial state per step"]
    G --> M["'messages'<br/>per-token AIMessageChunks"]
    G --> C["'custom'<br/>writer({...}) events from nodes"]
```

## Checkpointing unlocks 4 capabilities

```mermaid
flowchart TB
    CP["checkpointer=MemorySaver()<br/>(or SqliteSaver, AsyncPostgresSaver)"] --> R[resume after crash]
    CP --> I[interrupt + resume<br/>human approval]
    CP --> T[time travel<br/>get_state_history]
    CP --> E[state editing<br/>update_state]
```

## Prebuilt vs hand-rolled ReAct (the decision)

```mermaid
flowchart TB
    Q{"Need..."} -->|"3+ node loop?"| HR[hand-roll]
    Q -->|"Send fan-out inside?"| HR
    Q -->|"state-shape transforms?"| HR
    Q -->|"Otherwise"| PB["create_react_agent<br/>(prebuilt)"]
```

## What you can build by the end of Module 3

A real interactive tool-using agent that streams its work to the UI, pauses for human approval on expensive operations, and can be rewound and replayed.
