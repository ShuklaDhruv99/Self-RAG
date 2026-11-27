import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document

from selfrag.bot.schemas import SupportLabel, UseLabel


class State(TypedDict):
    question: str

    # What we actually send to the vector retriever (rewritten on retry)
    retrieval_query: str
    need_retrieval: bool

    docs: list[Document]            # everything the retriever returned
    relevant_docs: list[Document]   # the chunks that passed ISREL
    context: str                    # formatted relevant_docs, passed to generation

    answer: str                     # latest generated answer

    # Post-generation verification (labels only; the graders return no more)
    issup: SupportLabel | None
    isuse: UseLabel | None

    # Loop limits, compared with settings.max_rewrites / max_generations inside the routers
    rewrite_count: int              # query rewrites so far
    generation_count: int           # answers generated in the current retrieval round

    # Trace of node names. The reducer appends, so a node returns {"steps": ["name"]}.
    steps: Annotated[list[str], operator.add]

    final_answer: str               # what the user gets (may be the fallback message)


def initial_state(question: str) -> State:
    """Every field set, so nodes can read state[...] without a KeyError."""
    return State(
        question=question,
        retrieval_query=question,
        need_retrieval=True,
        docs=[],
        relevant_docs=[],
        context="",
        answer="",
        issup=None,
        isuse=None,
        rewrite_count=0,
        generation_count=0,
        steps=[],
        final_answer="",
    )
