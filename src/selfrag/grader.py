import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from selfrag.schemas import RetrieveDecision, RelevanceGrade, SupportGrade, SupportLabel, UsefulnessGrade, UseLabel
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

def is_sup(question: str, answer: str, context: str) -> SupportLabel:
    try:
        chain = IS_SUP_PROMPT | get_llm().with_structured_output(SupportGrade)
        decision: SupportGrade = invoke_with_retry(chain, {'question': question, 'answer': answer, 'context': context})
        return decision.is_sup
    except Exception:
        logger.exception("is_sup failed, defaulting to no_support")
        return "no_support"

def is_use(question: str, answer: str) -> UseLabel:
    try:
        chain = IS_USE_PROMPT | get_llm().with_structured_output(UsefulnessGrade)
        decision: UsefulnessGrade = invoke_with_retry(chain, {'question': question, 'answer': answer})
        return decision.is_use
    except Exception:
        logger.exception("is_use failed, defaulting to not_useful")
        return "not_useful"

if __name__ == '__main__':
    from selfrag.logging_setup import setup_logging

    setup_logging()

    question = "How much does the Growth plan cost per user per month?"

    result = is_use(question=question, answer="The Growth plan is best suited for growing teams.")

    print("OK" if result == 'useful' else "Fail")