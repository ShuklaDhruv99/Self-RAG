import logging

from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.schemas import SupportGrade
from selfrag.bot.state import State

logger = logging.getLogger(__name__)

IS_SUP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You verify whether the ANSWER is supported by the CONTEXT. Judge support only; "
            "do not judge whether the answer is complete or helpful.\n"
            "is_sup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "- fully_supported: every claim in the ANSWER is stated in, or directly follows from, the CONTEXT. "
            "Rewording or summarising stated facts is fine, and an answer that says it does not know "
            "because the CONTEXT lacks the information is also fully_supported.\n"
            "- partially_supported: the core facts are supported, BUT the ANSWER also adds a fact, "
            "or a judgment such as 'best value' or 'ideal for', that the CONTEXT does not state.\n"
            "- no_support: the key claims of the ANSWER are not in the CONTEXT, or contradict it.\n\n"
            "Examples:\n"
            "- Context: 'Growth Plan - Price: INR 5,999 per user per month'. "
            "Answer: 'The Growth plan costs INR 5,999 per user per month.' -> fully_supported.\n"
            "- Same context. Answer: 'The Growth plan costs INR 5,999 per user per month, "
            "making it the best value for growing teams.' -> partially_supported.\n"
            "- Same context. Answer: 'The Growth plan includes a free trial.' -> no_support.\n\n"
            "Rules:\n"
            "- Do not use outside knowledge.\n"
            "- evidence: up to 3 short verbatim quotes from the CONTEXT that support the answer."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)


def check_support(state: State) -> dict:
    try:
        chain = IS_SUP_PROMPT | get_llm().with_structured_output(SupportGrade)
        decision: SupportGrade = invoke_with_retry(
            chain,
            {"question": state["question"], "answer": state["answer"], "context": state["context"]},
        )
        label = decision.is_sup
    except Exception:
        logger.exception("check_support failed, defaulting to no_support")
        label = "no_support"
    return {"issup": label, "steps": ["check_support"]}
