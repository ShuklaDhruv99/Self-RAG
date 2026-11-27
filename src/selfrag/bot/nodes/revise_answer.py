from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.state import State

REVISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You revise an answer so that it is fully supported by the CONTEXT.\n\n"
            "The previous answer contained claims, details or judgments that the CONTEXT does not state.\n"
            "Rules:\n"
            "- Use only facts explicitly stated in the CONTEXT. Remove everything else.\n"
            "- Do not add opinions such as 'best value' or 'ideal for'.\n"
            "- If the CONTEXT does not contain the answer, say you don't know.\n"
            "- Do not mention the context or that you are revising."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Previous answer:\n{answer}\n\n"
            "Context:\n{context}"
        ),
    ]
)


def revise_answer(state: State) -> dict:
    chain = REVISE_PROMPT | get_llm() | StrOutputParser()
    answer = invoke_with_retry(
        chain,
        {"question": state["question"], "answer": state["answer"], "context": state["context"]},
    )
    return {
        "answer": answer,
        "generation_count": state["generation_count"] + 1,
        "steps": ["revise_answer"],
    }
