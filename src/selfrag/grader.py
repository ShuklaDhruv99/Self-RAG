import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from selfrag.schemas import RetrieveDecision, RelevanceGrade
from selfrag.llm import get_llm, invoke_with_retry

logger = logging.getLogger(__name__)

RETRIEVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether a question needs information from the company's documents.\n\n"
            "Guidelines:\n"
            "- should_retrieve=True for anything about the company's products, plans, pricing, "
            "features, policies or support, including simple 'what is X' questions about them.\n"
            "- should_retrieve=False only for greetings, thanks, chit-chat, or clearly general "
            "knowledge unrelated to the company.\n"
            "- If unsure, choose True. Wrongly skipping retrieval is worse than an extra search."
        ),
        ("human", "Question: {question}"),
    ]
)

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


def should_retrieve(question: str) -> bool:
    try:
        chain = RETRIEVE_PROMPT | get_llm().with_structured_output(RetrieveDecision)
        decision: RetrieveDecision = invoke_with_retry(chain, {"question": question})
        return decision.should_retrieve
    except Exception:
        logger.exception("should_retrieve failed, defaulting to True")
        return True

def is_relevant(question: str, document: Document) -> bool:
    try:
        chain = IS_RELEVANT_PROMPT | get_llm().with_structured_output(RelevanceGrade)
        decision: RelevanceGrade = invoke_with_retry(chain, {'question': question, 'document': document.page_content})
        return decision.is_relevant
    except Exception:
        logger.exception("is_relevant failed, defaulting to True")
        return True

if __name__ == '__main__':
    from selfrag.logging_setup import setup_logging
    from selfrag.vectorstore import get_vectorstore

    setup_logging()
    store = get_vectorstore()
    question = "Which plan includes NexaSupport?"
    for doc, _ in store.similarity_search_with_score(question, k=4):
        print(f"p{doc.metadata['page']} relevant={is_relevant(question, doc)}  {doc.page_content[:70]!r}")