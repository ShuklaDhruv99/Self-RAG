from langgraph.graph import END, START, StateGraph

from selfrag.bot.nodes.check_relevance import check_relevance
from selfrag.bot.nodes.check_support import check_support
from selfrag.bot.nodes.check_usefulness import check_usefulness
from selfrag.bot.nodes.decide_retrieval import decide_retrieval
from selfrag.bot.nodes.generate_direct import generate_direct
from selfrag.bot.nodes.generate_from_context import generate_from_context
from selfrag.bot.nodes.no_answer_found import no_answer_found
from selfrag.bot.nodes.retrieve import retrieve
from selfrag.bot.nodes.revise_answer import revise_answer
from selfrag.bot.nodes.rewrite_question import rewrite_question
from selfrag.bot.state import State, initial_state
from selfrag.core.config import settings

# Worst case is several retrieval rounds of retrieve/grade/generate/check/revise,
# so the default LangGraph limit of 25 steps is too low.
RECURSION_LIMIT = 60


# -----------------------------
# Routers: pure functions of the state (no API calls), so they are easy to unit test
# -----------------------------
def route_after_decide(state: State) -> str:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def route_after_relevance(state: State) -> str:
    if state["relevant_docs"]:
        return "generate_from_context"
    if state["rewrite_count"] < settings.max_rewrites:
        return "rewrite_question"
    return "no_answer_found"


def route_after_support(state: State) -> str:
    if state["issup"] == "fully_supported":
        return "check_usefulness"
    if state["generation_count"] < settings.max_generations:
        return "revise_answer"
    # Revision budget used up: a partly supported answer may still go on, an unsupported one must not.
    if state["issup"] == "partially_supported":
        return "check_usefulness"
    return "no_answer_found"


def route_after_usefulness(state: State) -> str:
    if state["isuse"] == "useful":
        return "end"
    if state["rewrite_count"] < settings.max_rewrites:
        return "rewrite_question"
    return "no_answer_found"


# -----------------------------
# Graph
# -----------------------------
def build_graph():
    g = StateGraph(State)

    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("check_relevance", check_relevance)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("check_support", check_support)
    g.add_node("revise_answer", revise_answer)
    g.add_node("check_usefulness", check_usefulness)
    g.add_node("rewrite_question", rewrite_question)
    g.add_node("no_answer_found", no_answer_found)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "check_relevance")
    g.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("generate_from_context", "check_support")
    g.add_conditional_edges(
        "check_support",
        route_after_support,
        {
            "check_usefulness": "check_usefulness",
            "revise_answer": "revise_answer",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("revise_answer", "check_support")

    g.add_conditional_edges(
        "check_usefulness",
        route_after_usefulness,
        {
            "end": END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("rewrite_question", "retrieve")
    g.add_edge("no_answer_found", END)

    return g.compile()


def run_graph(question: str) -> State:
    """Run the full Self-RAG flow for one question and return the final state."""
    return build_graph().invoke(initial_state(question), config={"recursion_limit": RECURSION_LIMIT})


if __name__ == "__main__":
    import sys

    from selfrag.core.logging import setup_logging

    setup_logging()
    q = " ".join(sys.argv[1:]) or "How much does the Growth plan cost per user per month?"
    result = run_graph(q)
    print("steps:", " -> ".join(result["steps"]))
    print("answer:", result["final_answer"])
