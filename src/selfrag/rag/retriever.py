import logging

from selfrag.core.config import settings
from selfrag.core.logging import setup_logging
from selfrag.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

def get_retriever(k: int | None = None):
    vector_store = get_vectorstore()
    k = k if k is not None else settings.top_k
    logger.debug("Building retriever with k=%d", k)
    retriever = vector_store.as_retriever(search_kwargs={'k': k})
    return retriever

if __name__ == "__main__":
    setup_logging()
    questions = [
        "How much does the Growth plan cost per user per month?",
        "Which plan includes NexaSupport?",
        "How many queries per month does Starter allow?",
        "What discount does annual billing give?",
        "Does NexaChat support role-based access control?",
        "Does NexaChat integrate with Telegram?",
    ]

    retriever = get_retriever()
    store = get_vectorstore()

    for question in questions:
        logger.info("Q: %s", question)
        for rank, doc in enumerate(retriever.invoke(question), start=1):
            logger.info("  %d p%s %s", rank, doc.metadata["page"], doc.page_content[:80].replace("\n", " "))

        for doc, score in store.similarity_search_with_score(question, k=4):
            logger.info("  score %.3f (lower = closer) p%s", score, doc.metadata["page"])
