"""
============================================================================
Stage 8: Cycles & the `Command` primitive (the critic feedback loop)
============================================================================

Up to now every graph we built was a DAG (directed acyclic graph) - data
flowed forward, never back. That's enough for simple pipelines but it
can't express the most powerful agent pattern: SELF-CORRECTION.

In the capstone swarm, the Critic reads the Searchers' findings, spots
gaps, and says "go search these 3 follow-up questions." That sends us
back UPSTREAM to the orchestrator. A loop. Capped at N rounds so we
don't spin forever.

This stage teaches two ideas:

  Idea 1: CYCLES
  --------------
  LangGraph has no problem with edges that point "backwards." A node
  named `critic` can route to a node named `orchestrator` even though
  orchestrator runs earlier in the normal flow. LangGraph just walks
  the edges; "earlier" / "later" is something we impose mentally.

  The only thing we MUST add ourselves: a TERMINATION CONDITION. Without
  one, a cycle is an infinite loop. Termination is usually:
    (a) a counter in state (max_critic_rounds = 2), or
    (b) the LLM signaling "I'm done" via structured output.

  Idea 2: `Command`
  -----------------
  Until now, a node returned a state-update dict, and a SEPARATE
  routing function (`add_conditional_edges`) decided where to go next.
  That works, but for cycles it forces you to split the "what did I
  decide?" logic across two functions.

  `Command` lets a node return BOTH: a state update AND the next node.
  It looks like this:

      from langgraph.types import Command

      def critic_node(state) -> Command[Literal["orchestrator", "END"]]:
          ...
          if needs_more_research:
              return Command(
                  goto="orchestrator",
                  update={"critic_round": state["critic_round"] + 1,
                          "follow_up_questions": new_qs},
              )
          else:
              return Command(goto="__end__")  # or END

  Two things to notice:
    * `goto=` is the routing decision (one of the typed Literal values).
    * `update=` is the state update (same dict you'd have returned).
    * The Literal[...] in the return type isn't required, but it lets
      LangGraph validate the graph at compile time (catches typos in
      target node names) and gives you autocomplete.

  When you use `Command` for routing, you DON'T need to call
  `add_conditional_edges` for that node. The Command IS the edge.

Graph topology (mermaid):

    ```mermaid
    flowchart LR
        S([START]) --> O[orchestrator]
        O --> SE[searcher]
        SE --> C[critic]
        C -.Command goto<br/>'orchestrator'<br/>(insufficient,<br/>under round cap).-> O
        C -.Command goto<br/>'synthesize'.-> SY[synthesize]
        SY --> E([END])
    ```

`Command` vs `add_conditional_edges` (when to use which):

    ```mermaid
    flowchart TB
        subgraph CMD["Command (this stage)<br/>Node returns state update + next node"]
            CN[node] -->|"return Command(<br/>goto='X',<br/>update={...})"| CT[next node X]
        end
        subgraph CE["add_conditional_edges (Stage 7)<br/>Node returns state update only"]
            EN[node] -->|"return {state update}"| ER[router fn]
            ER -->|"returns 'X'"| EX[next node X]
        end
    ```

The two termination patterns (you NEED at least one):

    ```mermaid
    flowchart LR
        subgraph Counter["Pattern 1: counter in state"]
            C1["if critic_round >= 2:<br/>force exit"]
        end
        subgraph LLMSig["Pattern 2: LLM signal"]
            L1["if verdict.sufficient:<br/>exit"]
        end
        Counter --> Both["use BOTH<br/>(belt + suspenders)"]
        LLMSig --> Both
    ```

The critic's logic:
  - if findings don't address the original query AND rounds < 2: loop
  - else: end

This is exactly the pattern in PROJECT2_PLAN.md section 4 ("Critic→
Searcher self-correction loop, capped at 2 rounds").
============================================================================
"""

import os
from typing import TypedDict, Annotated, Literal
from operator import add

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
# NEW: Command is the unified "state update + routing" primitive.
from langgraph.types import Command


# ---------------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing from .env"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY missing from .env"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tavily = TavilySearch(max_results=2, search_depth="basic")

MAX_CRITIC_ROUNDS = 2  # the kill-switch from PROJECT2_PLAN.md


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
# Two new fields compared to Stage 6:
#   * critic_round       - counter for the loop. Starts at 0, increments
#                          each time we go through the critic.
#   * pending_questions  - the list the orchestrator should search NEXT.
#                          Initially this is set from the original query;
#                          on subsequent loops the critic refills it.
#
# Note: pending_questions has NO reducer. It's a queue we OVERWRITE
# each round on purpose - the critic's new questions REPLACE the old.
# (In contrast, `findings` has the `add` reducer because we want
# findings from every round to accumulate.)
class ResearchState(TypedDict):
    query: str
    pending_questions: list[str]
    findings: Annotated[list[str], add]
    critic_round: int
    final_answer: str


# ---------------------------------------------------------------------------
# 2. ORCHESTRATOR - turns the query into questions to search
# ---------------------------------------------------------------------------
# On round 0 it expands the user's query into 2-3 sub-questions.
# On later rounds, `pending_questions` was already filled by the
# critic, so the orchestrator just passes them through.
class SubQuestions(BaseModel):
    questions: list[str] = Field(
        description="2-3 focused sub-questions answerable by web search."
    )

planner_llm = llm.with_structured_output(SubQuestions)

def orchestrator_node(state: ResearchState) -> dict:
    # If pending_questions is already populated (we're in a critic loop),
    # don't re-plan. Just announce and pass through.
    if state.get("pending_questions"):
        print(f"[orchestrator] round {state['critic_round']}: "
              f"using {len(state['pending_questions'])} follow-up Qs from critic")
        return {}

    print(f"[orchestrator] round 0: planning sub-questions")
    result: SubQuestions = planner_llm.invoke(
        f"Decompose into 2-3 sub-questions: {state['query']}"
    )
    return {"pending_questions": result.questions}


# ---------------------------------------------------------------------------
# 3. SEARCHER - runs Tavily on each pending question (sequentially for now)
# ---------------------------------------------------------------------------
# We could fan-out via Send (Stage 6) but for clarity we keep it
# sequential here - the focus of this stage is the LOOP, not parallelism.
# In the capstone we combine Stage 6's Send + Stage 8's Command.
def searcher_node(state: ResearchState) -> dict:
    qs = state["pending_questions"]
    print(f"[searcher] running {len(qs)} searches")
    new_findings = []
    for q in qs:
        raw = tavily.invoke({"query": q})
        for r in raw.get("results", []):
            new_findings.append(
                f"Q: {q}\n  -> {r.get('content', '')[:200]}"
            )
    # Clear pending_questions so on the next round the orchestrator
    # knows whether to plan or use critic-supplied questions.
    return {
        "findings": new_findings,
        "pending_questions": [],
    }


# ---------------------------------------------------------------------------
# 4. CRITIC - the cycle's brain (Command-based routing)
# ---------------------------------------------------------------------------
# The critic reads everything we've found and decides:
#   * If findings are sufficient -> goto "synthesize" (forward)
#   * Else if under round limit  -> goto "orchestrator" (BACKWARDS = cycle)
#   * Else                       -> goto "synthesize" (forced exit)
#
# Returns a `Command` carrying both the routing decision AND the state
# updates needed for that route (e.g. incrementing the round counter,
# stuffing new follow-up questions into pending_questions).
class CriticVerdict(BaseModel):
    sufficient: bool = Field(
        description="True if the existing findings adequately answer the "
                    "user's original query. False if there are clear gaps."
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="If not sufficient: 1-3 NEW sub-questions to research "
                    "that would close the gap. Empty list if sufficient.",
    )

critic_llm = llm.with_structured_output(CriticVerdict)

def critic_node(
    state: ResearchState,
) -> Command[Literal["orchestrator", "synthesize"]]:
    round_num = state["critic_round"]
    print(f"[critic] reviewing findings (round {round_num})")

    findings_block = "\n".join(state["findings"]) or "(empty)"
    verdict: CriticVerdict = critic_llm.invoke(
        "You are a research critic. Decide if the findings sufficiently "
        "answer the original query.\n\n"
        f"Original query: {state['query']}\n\n"
        f"Findings so far:\n{findings_block}"
    )

    # Termination conditions (in order):
    # 1. Critic says we have enough -> stop
    # 2. We've hit the round cap   -> stop (forced)
    # 3. Otherwise                  -> loop with new questions
    if verdict.sufficient:
        print(f"[critic] -> SUFFICIENT, going to synthesize")
        return Command(goto="synthesize")

    if round_num + 1 >= MAX_CRITIC_ROUNDS:
        print(f"[critic] -> round cap hit ({MAX_CRITIC_ROUNDS}), forcing exit")
        return Command(goto="synthesize")

    print(f"[critic] -> insufficient, looping with "
          f"{len(verdict.follow_up_questions)} follow-ups")
    # THE CYCLE: goto an UPSTREAM node, with state updates.
    return Command(
        goto="orchestrator",
        update={
            "pending_questions": verdict.follow_up_questions,
            "critic_round": round_num + 1,
        },
    )


# ---------------------------------------------------------------------------
# 5. SYNTHESIZER - final answer
# ---------------------------------------------------------------------------
def synthesize_node(state: ResearchState) -> dict:
    print(f"[synthesize] {len(state['findings'])} findings -> answer")
    findings_block = "\n\n".join(state["findings"]) or "(no findings)"
    response = llm.invoke(
        "Write a concise (4-6 sentence) answer using ONLY these findings.\n\n"
        f"Query: {state['query']}\n\nFindings:\n{findings_block}"
    )
    return {"final_answer": response.content}


# ---------------------------------------------------------------------------
# 6. WIRE THE GRAPH
# ---------------------------------------------------------------------------
# The cycle:  orchestrator -> searcher -> critic --(loop)--> orchestrator
#                                                  \--> synthesize -> END
#
# IMPORTANT: because critic routes via Command, we do NOT add
# conditional edges from `critic`. The Command IS the edge.
# We only need plain `add_edge`s for the deterministic transitions.
builder = StateGraph(ResearchState)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("searcher",     searcher_node)
builder.add_node("critic",       critic_node)
builder.add_node("synthesize",   synthesize_node)

builder.add_edge(START, "orchestrator")
builder.add_edge("orchestrator", "searcher")
builder.add_edge("searcher", "critic")
# NO edge needed from "critic" - Command handles that.
builder.add_edge("synthesize", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 7. RUN
# ---------------------------------------------------------------------------
# Try a query that probably needs follow-up to see the loop fire.
# Watch for "[critic] -> insufficient, looping" in the output.
if __name__ == "__main__":
    initial: ResearchState = {
        "query": "Compare LangGraph, AutoGen, and CrewAI - which has the best support for parallel agent execution and what are the trade-offs?",
        "pending_questions": [],
        "findings": [],
        "critic_round": 0,
        "final_answer": "",
    }
    final = graph.invoke(initial)

    print("\n=== RESULT ===")
    print(f"critic rounds used: {final['critic_round']}")
    print(f"total findings:     {len(final['findings'])}")
    print(f"\n{final['final_answer']}")
