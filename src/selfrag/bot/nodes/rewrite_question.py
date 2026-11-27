import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.state import State

logger = logging.getLogger(__name__)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6-16 words) and keep the original meaning.\n"
            "- Preserve key entities exactly (product names, plan names such as NexaAI, Growth, NexaSupport).\n"
            "- Remove filler and chatter, fix spelling, and expand vague wording into specific terms "
            "closely tied to what the question asks.\n"
            "- Do NOT invent product names, plan names, facts or numbers. Never turn a generic word "
            "(such as 'teams', 'basic' or 'premium') into a plan or product name; "
            "use only names that appear in the question.\n"
            "- Do NOT answer the question.\n"
            "- If a previous query is given, it did not retrieve useful chunks: write a query with "
            "different wording or a different angle, not a copy of it.\n"
            "- Output ONLY the query text, with no quotes, labels or explanation.\n\n"
            "Examples:\n"
            "Question: 'what about the pricing thing for teams?'\n"
            "Query: NexaAI pricing plans price per user per month\n\n"
            "Question: 'does nexachat do telegram?'\n"
            "Query: NexaChat Telegram integration supported integrations"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Previous query (none if empty):\n{previous_query}"
        )
    ]
)


def rewrite_question(state: State) -> dict:
    question, previous_query = state["question"], state["retrieval_query"]
    try:
        chain = REWRITE_PROMPT | get_llm() | StrOutputParser()
        rewritten = invoke_with_retry(chain, {"question": question, "previous_query": previous_query or "none"})
        new_query = rewritten.strip().strip('"\'') or question
    except Exception:
        logger.exception("rewrite_question failed, keeping the original question")
        new_query = question

    return {
        "retrieval_query": new_query,
        "rewrite_count": state["rewrite_count"] + 1,
        "generation_count": 0,  # each new retrieval round gets its own revision budget
        "steps": ["rewrite_question"],
    }
