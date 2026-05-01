"""
============================================================================
Stage 4 + Stage 5 combined: Real LLM call + Real Tool call
============================================================================

What you'll learn here:

  Stage 4 - Real LLM (Gemini via LangChain):
    * How to load API keys from .env
    * How to instantiate a Gemini chat model
    * How to get STRUCTURED output (a list of strings) instead of a blob
      of text - this is the single most useful LangChain feature for agents
    * How to plug an LLM call into a LangGraph node

  Stage 5 - Real Tool (Tavily web search):
    * What "tools" actually are in LangChain (just functions with metadata)
    * Calling a tool DIRECTLY from a node (the simple way - good for now)
    * The difference vs. "tool-calling agents" where the LLM picks the tool
      (we'll use that pattern later for the Searcher agent)

Graph shape:
    START -> planner (LLM) -> searcher (Tavily) -> summarizer (LLM) -> END

This is the same shape as Stage 2/3, but every node now does real work.
============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# Standard library
import os
from typing import TypedDict, Annotated
from operator import add

# 3rd party
from dotenv import load_dotenv          # reads .env -> os.environ
from pydantic import BaseModel, Field   # for STRUCTURED output schemas

# LangChain - the model + tool wrappers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# LangGraph - the orchestration layer
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 0. LOAD ENVIRONMENT
# ---------------------------------------------------------------------------
# load_dotenv() reads the .env file at the project root and sets each
# KEY=VALUE line as an environment variable for this Python process.
# After this line, os.getenv("GOOGLE_API_KEY") returns your real key.
#
# WHY .env: keeps secrets out of source code / git. The libraries
# (ChatGoogleGenerativeAI, TavilySearch) automatically look for their
# keys in os.environ, so you usually don't even need to pass them.
load_dotenv()

# Sanity check - if these are missing we want to fail fast with a clear msg.
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"


# ---------------------------------------------------------------------------
# 1. INSTANTIATE THE LLM (Gemini)
# ---------------------------------------------------------------------------
# This is a LangChain "chat model" object. Once created, it has a uniform
# interface (.invoke, .stream, .with_structured_output, ...) regardless
# of the underlying provider. That uniformity is the main reason to use
# LangChain at all - swap "ChatGoogleGenerativeAI" for "ChatOpenAI" and
# the rest of your code is unchanged.
#
# Model name notes:
#   * The PROJECT2_PLAN.md targets "gemini-3.1-pro-preview" / "gemini-3-flash-preview"
#     which are the cutting-edge 2026 models.
#   * For learning we use a stable, widely-available model first. We can
#     swap this string later without touching anything else.
#   * temperature=0 = deterministic-ish output, good for structured tasks
#     like decomposing a query into sub-questions where we don't want creativity.
LLM_MODEL = "gemini-3.1-pro-preview"   # fast + cheap, perfect for tutorial

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL
)


# ---------------------------------------------------------------------------
# 2. INSTANTIATE THE TOOL (Tavily search)
# ---------------------------------------------------------------------------
# A LangChain "tool" is just a callable object with a name, a description,
# and a typed input schema. The TavilySearch object wraps Tavily's REST API
# behind that uniform interface.
#
# Two ways to use a tool:
#   (a) Call it directly from Python: tavily.invoke({"query": "..."})
#       <- We do this in Stage 5 because it's simpler.
#   (b) Bind it to an LLM and let the LLM decide when to call it
#       (`llm.bind_tools([tavily])`). The LLM emits a "tool call" message
#       and a separate ToolNode actually executes it.
#       <- We'll use this later when we build the real Searcher agent.
#
# Parameters explained:
#   max_results=3        -> keep token cost low for the tutorial
#   search_depth="basic" -> faster + cheaper than "advanced"
#                          (advanced does a deeper crawl per result)
tavily = TavilySearch(
    max_results=3,
    search_depth="basic",
)


# ---------------------------------------------------------------------------
# 3. STATE
# ---------------------------------------------------------------------------
# Same shape as Stage 2/3, with reducers on the list fields so multiple
# writers (we'll have them in Stage 6) don't clobber each other.
class ResearchState(TypedDict):
    query: str
    sub_questions: Annotated[list[str], add]
    findings: Annotated[list[str], add]
    final_summary: str


# ---------------------------------------------------------------------------
# 4. PYDANTIC SCHEMA FOR STRUCTURED OUTPUT
# ---------------------------------------------------------------------------
# THE SINGLE MOST IMPORTANT TRICK IN THIS FILE.
#
# Problem: if we just do llm.invoke("decompose this query"), we get back
# free-form text like "Sure! Here are some sub-questions: 1. ... 2. ..."
# Now we have to parse that with regex / string splitting. Brittle. Ugly.
#
# Solution: define a Pydantic model describing the EXACT shape of the data
# we want, then call llm.with_structured_output(SubQuestions). Under the
# hood LangChain converts the Pydantic model into a JSON schema, asks the
# LLM to emit JSON matching that schema (using the provider's native
# function-calling / structured-output mode), and parses the result back
# into a Pydantic instance. We get a typed Python object out, every time.
#
# Field(...) with a description is critical: that description gets sent
# to the model as part of the schema, so it's effectively a per-field
# prompt. Use it to guide the model.
class SubQuestions(BaseModel):
    """A decomposition of a research query into focused sub-questions."""
    questions: list[str] = Field(
        description=(
            "3 to 5 focused sub-questions that together would answer the "
            "user's original query. Each must be self-contained and "
            "answerable by a web search."
        )
    )


# A version of the LLM bound to that schema. Calling .invoke on this
# returns a SubQuestions instance, not a string.
planner_llm = llm.with_structured_output(SubQuestions)

# ---------------------------------------------------------------------------
# 5. NODE: PLANNER (real LLM call w/ structured output)
# ---------------------------------------------------------------------------
# Replaces the hardcoded planner from Stage 2/3 with a real Gemini call.
def planner_node(state: ResearchState) -> dict:
    query = state["query"]
    print(f"[planner] decomposing query with Gemini: {query!r}")

    # The prompt is a plain string. LangChain accepts strings, lists of
    # message tuples like [("system", "..."), ("user", "...")], or
    # explicit Message objects. We use a simple string here because
    # the structured-output schema does most of the heavy lifting.
    prompt = (
        "You are a research planner. Decompose the following research query "
        "into 3 to 5 focused, self-contained sub-questions that can each be "
        "answered with a single web search.\n\n"
        f"Query: {query}"
    )

    # planner_llm.invoke() returns a SubQuestions instance directly.
    # No JSON parsing, no regex. This is why we use structured output.
    result: SubQuestions = planner_llm.invoke(prompt)

    print(f"[planner] got {len(result.questions)} sub-questions")
    return {"sub_questions": result.questions}


# ---------------------------------------------------------------------------
# 6. NODE: SEARCHER (real tool call)
# ---------------------------------------------------------------------------
# For now: take ONLY the first sub-question and run one Tavily search.
# In Stage 6 we'll fan this out and run a Searcher per sub-question
# in parallel using the Send API.
def searcher_node(state: ResearchState) -> dict:
    sub_qs = state["sub_questions"]
    if not sub_qs:
        print("[searcher] no sub-questions, skipping")
        return {}

    sub_q = sub_qs[0]
    print(f"[searcher] tavily search for: {sub_q!r}")

    # tavily.invoke takes a dict; "query" is the only required key.
    # The return shape is a dict like:
    #   {"query": "...", "results": [{"title", "url", "content", ...}, ...]}
    raw = tavily.invoke({"query": sub_q})
    results = raw.get("results", [])

    # Compress each result down to a short string so the next node has
    # something compact to summarize. In the real Searcher agent the LLM
    # will write the summary; here we just stitch raw snippets so you
    # can see the data flow without burning more tokens.
    findings = [
        f"[{r.get('title', '?')}] {r.get('content', '')[:200]}"
        for r in results
    ]
    print(f"[searcher] got {len(findings)} results")
    return {"findings": findings}


# ---------------------------------------------------------------------------
# 7. NODE: SUMMARIZER (real LLM call, free-form output)
# ---------------------------------------------------------------------------
# Demonstrates the OTHER kind of LLM call: just .invoke() with a prompt
# and use the .content of the AI message. No structured output needed
# because the answer is naturally prose.
def summarizer_node(state: ResearchState) -> dict:
    query = state["query"]
    findings = state["findings"]
    print(f"[summarizer] summarizing {len(findings)} findings with Gemini")

    # Joining findings with newlines is the simplest "context" you can
    # build. In the real Synthesizer agent we'll be much more careful
    # (e.g. include source ids, tag untrusted content, etc.).
    findings_block = "\n\n".join(f"- {f}" for f in findings) or "(no findings)"

    prompt = (
        "You are a research summarizer. Given the following findings from "
        "web search, write a concise (3-5 sentence) answer to the user's "
        "original query. Do not invent facts beyond the findings.\n\n"
        f"Original query: {query}\n\n"
        f"Findings:\n{findings_block}"
    )

    # llm.invoke() returns an AIMessage object. The text is in .content.
    # (When we build the production Synthesizer we'll switch to
    # structured output so we can extract per-claim citations.)
    response = llm.invoke(prompt)
    return {"final_summary": response.content}


# ---------------------------------------------------------------------------
# 8. WIRE THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("searcher", searcher_node)
builder.add_node("summarizer", summarizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "searcher")
builder.add_edge("searcher", "summarizer")
builder.add_edge("summarizer", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 9. RUN
# ---------------------------------------------------------------------------
# Notes on cost / quotas:
#   * One run of this graph makes ~2 Gemini calls + 1 Tavily search.
#   * On gemini-2.5-flash this is well under a cent.
#   * If you hit a 429 rate-limit error, just wait 30s and rerun.
if __name__ == "__main__":
    initial_state: ResearchState = {
        "query": "What is LangGraph and why is it used for multi-agent systems?",
        "sub_questions": [],
        "findings": [],
        "final_summary": "",
    }

    final_state = graph.invoke(initial_state)

    print("\n=== FINAL STATE ===")
    print("query:        ", final_state["query"])
    print("\nsub_questions:")
    for q in final_state["sub_questions"]:
        print("  -", q)
    print("\nfindings (truncated):")
    for f in final_state["findings"]:
        print("  -", f[:160], "...")
    print("\nfinal_summary:")
    print(final_state["final_summary"])
