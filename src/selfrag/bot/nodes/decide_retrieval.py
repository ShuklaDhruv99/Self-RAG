import logging

from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.schemas import RetrieveDecision
from selfrag.bot.state import State

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


def decide_retrieval(state: State) -> dict:
    try:
        chain = RETRIEVE_PROMPT | get_llm().with_structured_output(RetrieveDecision)
        decision: RetrieveDecision = invoke_with_retry(chain, {"question": state["question"]})
        need_retrieval = decision.should_retrieve
    except Exception:
        logger.exception("decide_retrieval failed, defaulting to True")
        need_retrieval = True
    return {"need_retrieval": need_retrieval, "steps": ["decide_retrieval"]}
