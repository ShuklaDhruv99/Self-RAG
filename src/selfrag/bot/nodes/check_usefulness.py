import logging

from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.schemas import UsefulnessGrade
from selfrag.bot.state import State

logger = logging.getLogger(__name__)

IS_USE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether an ANSWER is useful for the QUESTION.\n"
            "is_use must be one of: useful, not_useful.\n\n"
            "Judge only whether the answer addresses what the question asks. "
            "Do NOT judge whether it is factually correct or supported by documents; "
            "that is checked separately.\n\n"
            "- useful: the answer directly and specifically addresses the question and gives "
            "the user what they asked for.\n"
            "- not_useful: the answer is off-topic, vague, evasive, only repeats the question, "
            "answers a different question, or says it does not know.\n\n"
            "Examples:\n"
            "- Question: 'How much does the Growth plan cost per user per month?' / "
            "Answer: 'The Growth plan costs INR 5,999 per user per month.' -> useful.\n"
            "- Question: 'How much does the Growth plan cost per user per month?' / "
            "Answer: 'The Growth plan is best suited for growing teams.' -> not_useful "
            "(does not give the price).\n"
            "- Question: 'Does NexaChat integrate with Telegram?' / "
            "Answer: 'NexaChat is an internal knowledge assistant.' -> not_useful "
            "(does not answer the question).\n"
            "- Question: 'Does NexaAI offer a free trial?' / "
            "Answer: 'I don't know.' -> not_useful.\n\n"
            "Give a one-line reason first, then the verdict."
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)


def check_usefulness(state: State) -> dict:
    try:
        chain = IS_USE_PROMPT | get_llm().with_structured_output(UsefulnessGrade)
        decision: UsefulnessGrade = invoke_with_retry(chain, {"question": state["question"], "answer": state["answer"]})
        label = decision.is_use
    except Exception:
        logger.exception("check_usefulness failed, defaulting to not_useful")
        label = "not_useful"

    update = {"isuse": label, "steps": ["check_usefulness"]}
    if label == "useful":
        update["final_answer"] = state["answer"]
    return update
