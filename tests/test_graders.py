"""Tests for the grader schemas and graders.

Run everything:            pytest
Skip live Gemini calls:    pytest -m "not integration"
Run only live Gemini:      pytest -m integration
"""
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from selfrag import grader
from selfrag.grader import is_relevant, is_sup, is_use, rewrite_query, should_retrieve
from selfrag.schemas import (
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
def test_should_retrieve_skips_greeting():
    assert should_retrieve("hi") is False


@pytest.mark.integration
def test_should_retrieve_for_product_question():
    assert should_retrieve("How much is the Growth plan?") is True


@pytest.mark.integration
def test_is_relevant_yes():
    assert is_relevant(Q_PRICE, GROWTH_CHUNK) is True


@pytest.mark.integration
def test_is_relevant_no():
    assert is_relevant(Q_PRICE, BILLING_CHUNK) is False


@pytest.mark.integration
def test_is_sup_fully_supported():
    answer = "The Growth plan costs INR 5,999 per user per month."
    assert is_sup(Q_PRICE, answer, GROWTH_CONTEXT) == "fully_supported"


@pytest.mark.integration
def test_is_sup_no_support():
    answer = "NexaChat integrates with Telegram and Discord."
    assert is_sup(Q_PRICE, answer, GROWTH_CONTEXT) == "no_support"


@pytest.mark.integration
def test_is_use_useful():
    answer = "The Growth plan costs INR 5,999 per user per month."
    assert is_use(Q_PRICE, answer) == "useful"


@pytest.mark.integration
def test_is_use_not_useful_for_vague_answer():
    answer = "The Growth plan is best suited for growing teams."
    assert is_use(Q_PRICE, answer) == "not_useful"


@pytest.mark.integration
def test_is_use_not_useful_for_dont_know():
    assert is_use(Q_TRIAL, "I don't know.") == "not_useful"


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------

def test_rewrite_query_falls_back_to_original_question(monkeypatch):
    """If the LLM call fails, the loop should continue with the original question (no network)."""
    def boom(*args, **kwargs):
        raise RuntimeError("Gemini unavailable")

    monkeypatch.setattr(grader, "invoke_with_retry", boom)
    assert rewrite_query("what about pricing?") == "what about pricing?"


def test_rewrite_query_cleans_quotes_and_whitespace(monkeypatch):
    monkeypatch.setattr(grader, "invoke_with_retry", lambda chain, inputs: '  "NexaAI pricing plans"  \n')
    assert rewrite_query("pricing?") == "NexaAI pricing plans"


def test_rewrite_query_empty_output_returns_original(monkeypatch):
    monkeypatch.setattr(grader, "invoke_with_retry", lambda chain, inputs: "   ")
    assert rewrite_query("pricing?") == "pricing?"


@pytest.mark.integration
def test_rewrite_query_keeps_key_entities():
    rewritten = rewrite_query(Q_PRICE)
    assert "growth" in rewritten.lower()


@pytest.mark.integration
def test_rewrite_query_does_not_invent_plan_names():
    rewritten = rewrite_query("what about the pricing thing for teams?")
    assert "team plan" not in rewritten.lower()


@pytest.mark.integration
def test_rewrite_query_with_previous_query_changes_wording():
    question = "what about the pricing thing for teams?"
    first = rewrite_query(question)
    second = rewrite_query(question, previous_query=first)
    assert second.strip().lower() != first.strip().lower()
