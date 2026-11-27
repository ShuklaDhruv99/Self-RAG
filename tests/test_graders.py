"""Tests for the grader schemas and graders.

Run everything:            pytest
Skip live Gemini calls:    pytest -m "not integration"
Run only live Gemini:      pytest -m integration
"""
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

import selfrag.bot.nodes.rewrite_question as rewrite_module
from selfrag.bot.nodes.check_relevance import check_relevance
from selfrag.bot.nodes.check_support import check_support
from selfrag.bot.nodes.check_usefulness import check_usefulness
from selfrag.bot.nodes.decide_retrieval import decide_retrieval
from selfrag.bot.nodes.rewrite_question import rewrite_question
from selfrag.bot.state import initial_state
from selfrag.bot.schemas import (
    RelevanceGrade,
    RetrieveDecision,
    SupportGrade,
    UsefulnessGrade,
)

# ---------------------------------------------------------------------------
# Fixtures: fixed text copied from Product_and_Pricing.pdf, so tests need no DB
# ---------------------------------------------------------------------------

Q_PRICE = "How much does the Growth plan cost per user per month?"
Q_TRIAL = "Does NexaAI offer a free trial?"

GROWTH_CHUNK = Document(
    page_content=(
        "Growth Plan\n"
        "Best suited for growing teams requiring deeper insights and integrations.\n"
        "- Price: INR 5,999 per user per month\n"
        "- Access to NexaChat, NexaInsight, and NexaSupport\n"
        "- Priority email and chat support"
    ),
    metadata={"source": "Product_and_Pricing.pdf", "page": 3},
)

BILLING_CHUNK = Document(
    page_content=(
        "- Annual billing offers a 15% discount.\n"
        "- Invoices are generated monthly or annually based on billing preference.\n"
        "- Taxes are applicable as per local regulations."
    ),
    metadata={"source": "Product_and_Pricing.pdf", "page": 4},
)

GROWTH_CONTEXT = GROWTH_CHUNK.page_content


def state(**changes):
    s = initial_state(changes.get("question", Q_PRICE))
    s.update(changes)
    return s


# ---------------------------------------------------------------------------
# Schema tests (no network)
# ---------------------------------------------------------------------------

def test_retrieve_decision_valid():
    assert RetrieveDecision(should_retrieve=True).should_retrieve is True


def test_retrieve_decision_requires_field():
    with pytest.raises(ValidationError):
        RetrieveDecision()


def test_relevance_grade_valid():
    grade = RelevanceGrade(reason="Lists the price.", is_relevant=True)
    assert grade.is_relevant is True


def test_relevance_grade_requires_reason():
    with pytest.raises(ValidationError):
        RelevanceGrade(is_relevant=True)


def test_support_grade_accepts_valid_labels():
    for label in ("fully_supported", "partially_supported", "no_support"):
        assert SupportGrade(is_sup=label).is_sup == label


def test_support_grade_evidence_defaults_to_empty():
    assert SupportGrade(is_sup="no_support").evidence == []


def test_support_grade_rejects_unknown_label():
    with pytest.raises(ValidationError):
        SupportGrade(is_sup="maybe")


def test_usefulness_grade_valid():
    assert UsefulnessGrade(reason="Gives the price.", is_use="useful").is_use == "useful"


def test_usefulness_grade_rejects_unknown_label():
    with pytest.raises(ValidationError):
        UsefulnessGrade(reason="x", is_use="kind_of_useful")


# ---------------------------------------------------------------------------
# Integration tests (call the live Gemini API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_decide_retrieval_skips_greeting():
    assert decide_retrieval(state(question="hi"))["need_retrieval"] is False


@pytest.mark.integration
def test_decide_retrieval_for_product_question():
    assert decide_retrieval(state(question="How much is the Growth plan?"))["need_retrieval"] is True


@pytest.mark.integration
def test_check_relevance_keeps_relevant_chunk():
    update = check_relevance(state(question=Q_PRICE, docs=[GROWTH_CHUNK]))
    assert update["relevant_docs"] == [GROWTH_CHUNK]


@pytest.mark.integration
def test_check_relevance_drops_unrelated_chunk():
    update = check_relevance(state(question=Q_PRICE, docs=[BILLING_CHUNK]))
    assert update["relevant_docs"] == []


@pytest.mark.integration
def test_check_support_fully_supported():
    answer = "The Growth plan costs INR 5,999 per user per month."
    update = check_support(state(question=Q_PRICE, answer=answer, context=GROWTH_CONTEXT))
    assert update["issup"] == "fully_supported"


@pytest.mark.integration
def test_check_support_no_support():
    answer = "NexaChat integrates with Telegram and Discord."
    update = check_support(state(question=Q_PRICE, answer=answer, context=GROWTH_CONTEXT))
    assert update["issup"] == "no_support"


@pytest.mark.integration
def test_check_usefulness_useful():
    answer = "The Growth plan costs INR 5,999 per user per month."
    update = check_usefulness(state(question=Q_PRICE, answer=answer))
    assert update["isuse"] == "useful"
    assert update["final_answer"] == answer


@pytest.mark.integration
def test_check_usefulness_not_useful_for_vague_answer():
    answer = "The Growth plan is best suited for growing teams."
    assert check_usefulness(state(question=Q_PRICE, answer=answer))["isuse"] == "not_useful"


@pytest.mark.integration
def test_check_usefulness_not_useful_for_dont_know():
    assert check_usefulness(state(question=Q_TRIAL, answer="I don't know."))["isuse"] == "not_useful"


# ---------------------------------------------------------------------------
# rewrite_question
# ---------------------------------------------------------------------------

def test_rewrite_question_falls_back_to_original_question(monkeypatch):
    """If the LLM call fails, the loop should continue with the original question (no network)."""
    def boom(*args, **kwargs):
        raise RuntimeError("Gemini unavailable")

    monkeypatch.setattr(rewrite_module, "invoke_with_retry", boom)
    update = rewrite_question(state(question="what about pricing?", retrieval_query="what about pricing?"))
    assert update["retrieval_query"] == "what about pricing?"


def test_rewrite_question_cleans_quotes_and_whitespace(monkeypatch):
    monkeypatch.setattr(rewrite_module, "invoke_with_retry", lambda chain, inputs: '  "NexaAI pricing plans"  \n')
    assert rewrite_question(state(question="pricing?"))["retrieval_query"] == "NexaAI pricing plans"


def test_rewrite_question_empty_output_returns_original(monkeypatch):
    monkeypatch.setattr(rewrite_module, "invoke_with_retry", lambda chain, inputs: "   ")
    assert rewrite_question(state(question="pricing?"))["retrieval_query"] == "pricing?"


@pytest.mark.integration
def test_rewrite_question_keeps_key_entities():
    update = rewrite_question(state(question=Q_PRICE, retrieval_query=Q_PRICE))
    assert "growth" in update["retrieval_query"].lower()


@pytest.mark.integration
def test_rewrite_question_does_not_invent_plan_names():
    question = "what about the pricing thing for teams?"
    update = rewrite_question(state(question=question, retrieval_query=question))
    assert "team plan" not in update["retrieval_query"].lower()


@pytest.mark.integration
def test_rewrite_question_with_previous_query_changes_wording():
    question = "what about the pricing thing for teams?"
    first = rewrite_question(state(question=question, retrieval_query=question))["retrieval_query"]
    second = rewrite_question(state(question=question, retrieval_query=first))["retrieval_query"]
    assert second.strip().lower() != first.strip().lower()
