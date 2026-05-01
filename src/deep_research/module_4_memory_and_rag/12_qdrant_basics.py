"""
============================================================================
Stage 14: Qdrant + embeddings - the raw fundamentals (no LangChain)
============================================================================

Before we build RAG (Stage 15+), we have to demystify what's actually
happening underneath. LangChain wraps everything in 3-letter acronyms
(VDB, LLM, RAG, etc.) and the abstractions hide a stack of simple
operations. So this stage is intentionally LANGCHAIN-FREE: we use the
raw `qdrant-client` and the raw Google `genai` SDK to:

    1. Turn text into a vector (an "embedding")
    2. Store those vectors in Qdrant (the local vector DB on :6333)
    3. Search by meaning (the SAME meaning -> nearby vectors)
    4. Filter by metadata (e.g. "only docs from 2025")
    5. Update / delete points (housekeeping)

Once these five operations are clear, EVERY RAG paper / framework
makes sense - they're all just composing these primitives.

The two-step mental model
-------------------------
              "What's the weather like?"
                       |
                       | (1) embed
                       v
            [0.013, -0.241, 0.882, ..., 0.005]      (3072 numbers)
                       |
                       | (2) cosine-similarity search in Qdrant
                       v
              [matching docs, sorted by relevance]

That's the whole game. Embeddings turn semantic meaning into geometric
distance. Vector DBs do fast nearest-neighbor search in 1000-D+ spaces.

Why Qdrant (vs Chroma, Pinecone, pgvector, ...)
-----------------------------------------------
* Open source, self-hostable (we run it on :6333 via docker)
* Excellent metadata filtering (built-in payload index)
* Rust-native, very fast at billions of vectors
* The capstone uses Qdrant because PROJECT2_PLAN.md mem0 config
  picks Qdrant as the vector store

Why Gemini's embedding model
----------------------------
* Same vendor as our chat LLM = one API key, consistent quotas
* `gemini-embedding-001` produces high-quality 768/1536/3072-D vectors
* The capstone uses output_dimensionality=3072 for max quality;
  for the tutorial we use 768 to keep storage tiny

What we'll build in this file
-----------------------------
A toy "knowledge base" of ~12 documents about LangGraph, Qdrant,
agents, RAG, and a few off-topic distractors. Then we'll:
  - Embed all 12
  - Upsert them into a Qdrant collection
  - Run 4 semantic queries with different filters
  - Inspect what comes back

The full pipeline (mermaid):

    ```mermaid
    flowchart TB
        subgraph Ingest["INGEST (one-time)"]
            D["doc text"] -->|"embed_batch<br/>task=DOCUMENT"| V["vectors"]
            D --> P["payload<br/>{topic, year, text}"]
            V --> PT[PointStruct]
            P --> PT
            PT --> Q[(Qdrant<br/>:6333)]
        end
        subgraph Query["QUERY (per request)"]
            UQ["user query"] -->|"embed_text<br/>task=QUERY"| QV["query vector"]
            QV --> QP["qdrant.query_points<br/>+ optional Filter"]
            QP --> Q
            Q --> HITS["top-K hits<br/>(score, payload)"]
        end
    ```

Asymmetric retrieval (the task_type gotcha):

    ```mermaid
    flowchart LR
        DT["doc text"] -->|"task_type='RETRIEVAL_DOCUMENT'"| DE["doc embedding"]
        QT["user question"] -->|"task_type='RETRIEVAL_QUERY'"| QE["query embedding"]
        DE -.same shared space.- QE
        DE --> CMP{cosine similarity}
        QE --> CMP
        CMP --> SCR[score]
    ```

Filtered semantic search (hard constraint + similarity):

    ```mermaid
    flowchart LR
        QV[query vector] --> NN[nearest-neighbor<br/>cosine search]
        F[Filter:<br/>topic='qdrant'] --> NN
        NN -->|narrowed candidate set| TOP[top-K]
    ```
============================================================================
"""

import os
import uuid

from dotenv import load_dotenv

# Raw Qdrant client - no LangChain in sight.
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

# Raw Gemini client for embeddings - again, no LangChain.
# We use google.genai (Google's official SDK) directly so you can see
# what an embedding API call actually looks like.
from google import genai
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"

# Connect to the local Qdrant. The user already has it running on
# port 6333 per the .env. host="localhost", port=6333 is the default.
qdrant = QdrantClient(host="localhost", port=6333)

# Gemini client. The SDK reads GOOGLE_API_KEY from the environment.
gemini = genai.Client()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # we explicitly request 768-D vectors below
COLLECTION = "applied_langgraph_tutorial_14"


# ---------------------------------------------------------------------------
# 1. PRIMITIVE: turn TEXT -> VECTOR
# ---------------------------------------------------------------------------
# An "embedding" is just a list of floats. The model is trained so that
# semantically similar texts produce vectors close together (by cosine
# similarity), and dissimilar texts produce far-apart vectors.
#
# Two task types matter:
#   "RETRIEVAL_DOCUMENT" - use when EMBEDDING THE STORED DOCS
#   "RETRIEVAL_QUERY"    - use when EMBEDDING THE USER'S QUESTION
# Using the right task type meaningfully improves retrieval quality
# because Gemini's embeddings are asymmetric - docs and queries are
# embedded into the same space but with slightly different objectives.
def embed_text(text: str, *, task_type: str) -> list[float]:
    """Return a single 768-D embedding for `text`."""
    response = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    # `response.embeddings` is a list (one per input). Single text -> [0].
    return response.embeddings[0].values


def embed_batch(texts: list[str], *, task_type: str) -> list[list[float]]:
    """Embed a batch in one API call - much cheaper than N single calls."""
    response = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [e.values for e in response.embeddings]


# ---------------------------------------------------------------------------
# 2. PRIMITIVE: create a Qdrant COLLECTION
# ---------------------------------------------------------------------------
# A "collection" is Qdrant's name for a table. You declare:
#   - vector size (must match the embedding dim)
#   - distance metric (COSINE for normalized embeddings)
# Once created, you upsert "points" (vector + id + metadata payload).
def ensure_collection():
    if qdrant.collection_exists(COLLECTION):
        # Wipe and recreate so the tutorial is repeatable.
        qdrant.delete_collection(COLLECTION)

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"[qdrant] (re)created collection {COLLECTION!r} "
          f"with dim={EMBED_DIM} distance=cosine")


# ---------------------------------------------------------------------------
# 3. PRIMITIVE: upsert documents (text -> embedding -> point)
# ---------------------------------------------------------------------------
# A "point" in Qdrant is:
#   id      - unique identifier (uuid or int)
#   vector  - the embedding
#   payload - arbitrary JSON metadata (used for filtering AND returned
#             alongside results so you can show source text)
def upsert_documents(docs: list[dict]):
    """`docs` is a list of {text, topic, year} dicts."""
    print(f"[qdrant] embedding + upserting {len(docs)} docs")
    vectors = embed_batch(
        [d["text"] for d in docs],
        task_type="RETRIEVAL_DOCUMENT",   # IMPORTANT: doc side
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text":  d["text"],
                "topic": d["topic"],
                "year":  d["year"],
            },
        )
        for d, vec in zip(docs, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"[qdrant] upserted. collection size: "
          f"{qdrant.count(COLLECTION, exact=True).count}")


# ---------------------------------------------------------------------------
# 4. PRIMITIVE: semantic search (with optional metadata filter)
# ---------------------------------------------------------------------------
# Two flavors of search you'll use 99% of the time:
#   * pure semantic: "find the K most similar"
#   * filtered semantic: "find the K most similar AMONG docs where year=2025"
#
# The filter narrows the candidate set BEFORE similarity scoring -
# it's a hard constraint, not a hint. Use it for hard rules
# ("only this user's docs", "only verified facts").
def semantic_search(
    query: str,
    *,
    k: int = 3,
    filter_topic: str | None = None,
    filter_min_year: int | None = None,
):
    qvec = embed_text(query, task_type="RETRIEVAL_QUERY")  # IMPORTANT: query side

    # Build a Qdrant filter only if the caller asked for one.
    must_conditions = []
    if filter_topic is not None:
        must_conditions.append(
            FieldCondition(key="topic", match=MatchValue(value=filter_topic))
        )
    qdrant_filter = Filter(must=must_conditions) if must_conditions else None

    # query_points is the modern API (replaces deprecated `search`).
    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    # Optional Python-side post-filter for `min_year` to demonstrate that
    # you can mix Qdrant filters with your own logic.
    hits = res.points
    if filter_min_year is not None:
        hits = [h for h in hits if h.payload.get("year", 0) >= filter_min_year]
    return hits


# ---------------------------------------------------------------------------
# 5. THE TUTORIAL CORPUS
# ---------------------------------------------------------------------------
# 12 short docs spanning a few topics + years. We'll search across them.
CORPUS = [
    {"topic": "langgraph", "year": 2024,
     "text": "LangGraph is a low-level orchestration framework for building stateful, multi-actor agent applications using a graph of nodes."},
    {"topic": "langgraph", "year": 2025,
     "text": "LangGraph's Send API enables fan-out: a single planner node can dispatch many worker nodes in parallel, each with its own input."},
    {"topic": "langgraph", "year": 2025,
     "text": "Reducers like Annotated[list, add] tell LangGraph how to merge concurrent state updates from parallel branches without losing data."},
    {"topic": "qdrant",    "year": 2024,
     "text": "Qdrant is an open-source vector database written in Rust, supporting dense vector search with rich payload filtering."},
    {"topic": "qdrant",    "year": 2025,
     "text": "Qdrant payload indices accelerate filtered search: combining vector similarity with structured filters runs in milliseconds at scale."},
    {"topic": "agents",    "year": 2025,
     "text": "Multi-agent research systems with specialized roles often outperform single-agent setups by reducing context pollution per worker."},
    {"topic": "agents",    "year": 2024,
     "text": "The ReAct pattern interleaves reasoning steps with tool calls; an LLM emits 'tool_calls' and a tool node executes them."},
    {"topic": "rag",       "year": 2025,
     "text": "Retrieval-Augmented Generation grounds LLM answers in a private knowledge base, reducing hallucination on domain-specific questions."},
    {"topic": "rag",       "year": 2025,
     "text": "Self-RAG grades each retrieved document, decides whether to re-search, and rewrites the query if retrieval was poor - a graph-shaped pipeline."},
    {"topic": "embeddings","year": 2025,
     "text": "Gemini's embedding model supports asymmetric retrieval: pass task_type='RETRIEVAL_DOCUMENT' for stored text, 'RETRIEVAL_QUERY' for user questions."},
    {"topic": "off_topic", "year": 2024,
     "text": "Sourdough bread relies on a wild yeast and bacteria starter for its tang and rise; flour, water, and salt are the only ingredients."},
    {"topic": "off_topic", "year": 2024,
     "text": "Saturn's hexagon is a stable jet stream around its north pole, observed by Voyager 1 and later Cassini."},
]


# ---------------------------------------------------------------------------
# 6. RUN
# ---------------------------------------------------------------------------
def pretty_hits(label: str, hits):
    print(f"\n--- {label} ---")
    for i, h in enumerate(hits, 1):
        score = h.score
        topic = h.payload.get("topic", "?")
        year  = h.payload.get("year", "?")
        text  = h.payload.get("text", "")[:90]
        print(f"  {i}. (score={score:.3f}, topic={topic}, year={year}) {text}...")


if __name__ == "__main__":
    ensure_collection()
    upsert_documents(CORPUS)

    # Query 1: pure semantic - sourdough should rank LAST despite the corpus
    # having two unrelated off-topic docs.
    pretty_hits(
        "Q1: 'how do I dispatch many parallel workers in LangGraph?'",
        semantic_search("how do I dispatch many parallel workers in LangGraph?", k=3),
    )

    # Query 2: filtered to topic=qdrant. We expect ONLY qdrant docs back.
    pretty_hits(
        "Q2: 'fast filtered search at scale' (topic=qdrant)",
        semantic_search("fast filtered search at scale", k=3, filter_topic="qdrant"),
    )

    # Query 3: ambiguous about agents - which doc wins?
    pretty_hits(
        "Q3: 'what's the ReAct loop'",
        semantic_search("what's the ReAct loop", k=3),
    )

    # Query 4: Python-side post-filter for year>=2025.
    pretty_hits(
        "Q4: 'critic loop in research swarms' (year>=2025)",
        semantic_search("critic loop in research swarms", k=5, filter_min_year=2025),
    )

    # ----------------------------------------------------------------------
    # What you should take away from this file
    # ----------------------------------------------------------------------
    # * Embedding = text -> vector (use task_type properly)
    # * Vector DB = a thing that does fast cosine search at scale
    # * Payload  = arbitrary metadata used for filters AND for showing
    #              the source text alongside hits
    # * RAG, Self-RAG, Agentic RAG are ALL just clever orchestrations of
    #   the operations in this file. Stage 15 wires them into LangGraph.
