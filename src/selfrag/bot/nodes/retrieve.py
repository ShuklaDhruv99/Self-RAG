from selfrag.bot.state import State
from selfrag.rag.retriever import get_retriever


def retrieve(state: State) -> dict:
    docs = get_retriever().invoke(state["retrieval_query"])
    return {"docs": docs, "steps": ["retrieve"]}
