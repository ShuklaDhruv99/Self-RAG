"""Router and node-wrapper tests. No network: nodes that call an LLM are monkeypatched."""
import pytest
from langchain_core.documents import Document

import selfrag.bot.nodes.check_relevance as check_relevance_module
import selfrag.bot.nodes.check_usefulness as check_usefulness_module
import selfrag.bot.nodes.rewrite_question as rewrite_module
from selfrag.bot.schemas import UsefulnessGrade
from selfrag.bot.graph import (
    build_graph,
    route_after_decide,
    route_after_relevance,
    route_after_support,
    route_after_usefulness,
)
from selfrag.bot.nodes.no_answer_found import NO_ANSWER_MESSAGE, no_answer_found
from selfrag.bot.state import initial_state
from selfrag.core.config import settings

DOC = Document(page_content="Growth Plan - INR 5,999", metadata={"source": "x.pdf", "page": 3})


def state(**changes):
    s = initial_state("How much is the Growth plan?")
    s.update(changes)
    return s


# --- state ------------------------------------------------------------------

def test_initial_state_has_every_field():
    s = initial_state("hi")
    assert s["question"] == "hi" and s["retrieval_query"] == "hi"
    assert s["rewrite_count"] == 0 and s["generation_count"] == 0 and s["steps"] == []


def test_graph_compiles():
    assert build_graph() is not None


# --- routers ----------------------------------------------------------------

def test_route_after_decide():
    assert route_after_decide(state(need_retrieval=True)) == "retrieve"
    assert route_after_decide(state(need_retrieval=False)) == "generate_direct"


def test_route_after_relevance_generates_when_docs_survive():
    assert route_after_relevance(state(relevant_docs=[DOC])) == "generate_from_context"


def test_route_after_relevance_rewrites_then_gives_up():
    assert route_after_relevance(state(relevant_docs=[], rewrite_count=0)) == "rewrite_question"
    assert route_after_relevance(state(relevant_docs=[], rewrite_count=settings.max_rewrites)) == "no_answer_found"


def test_route_after_support_fully_supported_moves_on():
    assert route_after_support(state(issup="fully_supported", generation_count=1)) == "check_usefulness"


def test_route_after_support_revises_until_budget_is_used():
    assert route_after_support(state(issup="partially_supported", generation_count=1)) == "revise_answer"
    assert route_after_support(state(issup="no_support", generation_count=1)) == "revise_answer"


def test_route_after_support_at_limit():
    at_limit = settings.max_generations
    assert route_after_support(state(issup="partially_supported", generation_count=at_limit)) == "check_usefulness"
    assert route_after_support(state(issup="no_support", generation_count=at_limit)) == "no_answer_found"


def test_route_after_usefulness():
    assert route_after_usefulness(state(isuse="useful")) == "end"
    assert route_after_usefulness(state(isuse="not_useful", rewrite_count=0)) == "rewrite_question"
    assert route_after_usefulness(state(isuse="not_useful", rewrite_count=settings.max_rewrites)) == "no_answer_found"


# --- node wrappers (LLM calls replaced) -------------------------------------

def test_no_answer_found_sets_fallback():
    update = no_answer_found(state())
    assert update["final_answer"] == NO_ANSWER_MESSAGE
    assert update["steps"] == ["no_answer_found"]


def test_check_relevance_keeps_only_relevant_docs(monkeypatch):
    other = Document(page_content="Billing info", metadata={"source": "x.pdf", "page": 4})
    monkeypatch.setattr(check_relevance_module, "_grade_chunk", lambda q, d: d is DOC)
    update = check_relevance_module.check_relevance(state(docs=[DOC, other]))
    assert update["relevant_docs"] == [DOC]
    assert "Growth Plan" in update["context"] and "Billing info" not in update["context"]


def test_check_relevance_with_no_docs_makes_no_calls(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not be called")
    monkeypatch.setattr(check_relevance_module, "_grade_chunk", boom)
    assert check_relevance_module.check_relevance(state(docs=[]))["relevant_docs"] == []


def test_check_usefulness_sets_final_answer_only_when_useful(monkeypatch):
    useful = UsefulnessGrade(reason="x", is_use="useful")
    monkeypatch.setattr(check_usefulness_module, "invoke_with_retry", lambda chain, inputs: useful)
    assert check_usefulness_module.check_usefulness(state(answer="A"))["final_answer"] == "A"

    not_useful = UsefulnessGrade(reason="x", is_use="not_useful")
    monkeypatch.setattr(check_usefulness_module, "invoke_with_retry", lambda chain, inputs: not_useful)
    assert "final_answer" not in check_usefulness_module.check_usefulness(state(answer="A"))


def test_rewrite_question_updates_counters(monkeypatch):
    monkeypatch.setattr(rewrite_module, "invoke_with_retry", lambda chain, inputs: "new query")
    update = rewrite_module.rewrite_question(state(rewrite_count=0, generation_count=2))
    assert update["retrieval_query"] == "new query"
    assert update["rewrite_count"] == 1 and update["generation_count"] == 0
