from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from selfrag.bot.llm import get_llm, invoke_with_retry
from selfrag.bot.state import State

DIRECT_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only your general knowledge.\n"
            "If it requires specific company info, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: State) -> dict:
    chain = DIRECT_GENERATION_PROMPT | get_llm() | StrOutputParser()
    answer = invoke_with_retry(chain, {"question": state["question"]})
    return {"answer": answer, "final_answer": answer, "steps": ["generate_direct"]}
