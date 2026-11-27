import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.schemas import RelevanceGrade
from selfrag.bot.state import State
from selfrag.rag.baseline import format_context

logger = logging.getLogger(__name__)

IS_RELEVANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether a document chunk is relevant to a question.\n\n"
            "A chunk is relevant if it contains information that helps answer the question, "
            "even if it holds only part of the answer. It does NOT need to answer the question fully.\n"
            "A chunk also counts as relevant if it states something that IMPLIES part of the answer, "
            "even without naming the entity in the question.\n"
            "A chunk that is merely about the same general topic, but has nothing that helps "
            "answer this specific question, is NOT relevant.\n\n"
            "Examples:\n"
            "- Question: 'Which plan includes NexaSupport?' / chunk listing a plan's included products "
            "-> relevant.\n"
            "- Question: 'Which plan includes NexaSupport?' / chunk saying a plan includes 'all products' "
            "-> relevant (all products implies NexaSupport).\n"
            "- Question: 'What is the Growth plan price?' / chunk with the Starter plan price only "
            "-> not relevant.\n"
            "- Question: 'Does NexaChat integrate with Telegram?' / chunk about billing discounts "
            "-> not relevant.\n\n"
            "When unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}")
    ]
)


def _grade_chunk(question: str, document: Document) -> bool:
    """ISREL for one chunk. Keeps the chunk if grading fails."""
    try:
        chain = IS_RELEVANT_PROMPT | get_llm().with_structured_output(RelevanceGrade)
        decision: RelevanceGrade = invoke_with_retry(chain, {"question": question, "document": document.page_content})
        return decision.is_relevant
    except Exception:
        logger.exception("relevance grading failed, keeping the chunk")
        return True


def check_relevance(state: State) -> dict:
    """Keep only the chunks that pass ISREL. Graded against the original question, not the rewritten query."""
    question, docs = state["question"], state["docs"]
    if not docs:
        return {"relevant_docs": [], "context": "", "steps": ["check_relevance"]}

    with ThreadPoolExecutor(max_workers=len(docs)) as pool:
        verdicts = list(pool.map(lambda d: _grade_chunk(question, d), docs))

    relevant = [d for d, keep in zip(docs, verdicts) if keep]
    return {
        "relevant_docs": relevant,
        "context": format_context(relevant),
        "steps": ["check_relevance"],
    }
