# Module 4 — Memory & RAG

The agent's eyes and ears: embeddings, vector search, and the family of retrieval-augmented patterns that turn an LLM into a researcher.

| File | Stage | Concepts |
|---|---|---|
| [`12_qdrant_basics.py`](12_qdrant_basics.py) | 14 | Embeddings (`gemini-embedding-001`), Qdrant collections, upsert + search, metadata filters — *no LangChain*, raw clients |
| [`13_naive_rag.py`](13_naive_rag.py) | 15 | 2-node RAG (`retrieve → generate`); tagged-chunk prompt-injection defense; citations by `source_id` |
| `14_better_rag.py` (planned) | 16 | Query rewriting, hybrid search, reranking |
| `15_self_rag.py` (planned) | 17 | Graded retrieval + re-search loop (graph-shaped RAG) |
| `16_agentic_rag.py` (planned) | 18 | Retriever-as-tool — LLM decides whether to retrieve |
| `17_long_term_memory.py` (planned) | 19 | LangGraph `BaseStore`, `InMemoryStore` → `PostgresStore`; mem0 |

---

## The 5 primitives every RAG paper composes

```mermaid
flowchart LR
    P1[embed text] --> P2[create collection]
    P2 --> P3[upsert points]
    P3 --> P4[query_points<br/>cosine top-K]
    P4 --> P5[Filter<br/>metadata constraints]
```

## The RAG family tree

```mermaid
flowchart TB
    NAIVE["Stage 15: Naive RAG<br/>retrieve -> generate"] --> BETTER["Stage 16: Better RAG<br/>+ query rewriting<br/>+ hybrid search<br/>+ reranking"]
    BETTER --> SELF["Stage 17: Self-RAG<br/>+ grade each retrieval<br/>+ re-search loop<br/>(uses Stage 8 cycles!)"]
    SELF --> AGENT["Stage 18: Agentic RAG<br/>retriever as a TOOL<br/>(uses Stage 11 ReAct!)"]
    AGENT --> MEM["Stage 19: Long-term memory<br/>cross-session memory<br/>(mem0 / BaseStore)"]
```

Each stage REUSES patterns from earlier modules:
- Self-RAG = Module 2 cycles applied to RAG
- Agentic RAG = Module 3 ReAct with a retriever tool
- Long-term memory = Module 3 checkpointing + cross-thread store

## End-to-end naive RAG (Stage 15)

```mermaid
flowchart TB
    UQ[user query] --> E1["embed(QUERY)"]
    E1 --> Q[(Qdrant)]
    Q --> CHK[top-K chunks]
    CHK --> WR["wrap each in<br/><retrieved_chunk source_id='X'>"]
    UQ --> P[prompt: 'use ONLY chunks,<br/>cite by source_id']
    WR --> P
    P --> LLM[Gemini]
    LLM --> A["answer with<br/>[source_id] citations"]
```

## Why we built Qdrant raw first (Stage 14)

```mermaid
flowchart LR
    RAW["Stage 14<br/>raw qdrant-client<br/>raw google.genai"] --> UND[understand the primitives]
    UND --> WRAP["Stages 15+<br/>LangChain-wrapped<br/>versions"]
    UND --> CAP[capstone<br/>can debug at any layer]
```

When LangChain's `Qdrant` wrapper does something weird (and it will), you'll know whether the bug is in your code, the wrapper, or Qdrant itself — because you've used Qdrant directly.
