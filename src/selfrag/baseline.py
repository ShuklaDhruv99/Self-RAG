from functools import lru_cache
import logging
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from selfrag.config import settings
from selfrag.logging_setup import setup_logging
from selfrag.retriever import get_retriever

logger = logging.getLogger(__name__)


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int

class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a business rag chatbot.\n\n"
        "You will receive a CONTEXT block from internal company documents.\n"
        "If the context does not contain the answer, say you don't know.\n"
        "Task:\n"
        "Answer the question based on the context.\n"
        "Don't mention that you are getting a context in your answer."
    ),
    ("human", "Question:\n{question}\n\nContext:\n{context}")
])


def retrieve(question: str, k: int | None = None) -> list[Document]:
    return get_retriever(k).invoke(question)

def format_context(docs: list[Document]):
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[{i}] ({d.metadata['source']}, p.{d.metadata['page']})\n{d.page_content}")
    return "\n\n".join(parts)

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        timeout=30,
        max_retries=2
    )

def generate(question: str, context: str) -> str:
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({'question':question,'context':context})

def answer(question: str) -> RAGResponse:
    docs = retrieve(question)
    context = format_context(docs)
    text = generate(question, context)
    sources = [SourceChunk(text=d.page_content, source=d.metadata['source'], page=d.metadata['page']) for d in docs]
    logger.info("answered with %d sources", len(sources))
    return RAGResponse(answer=text, sources=sources)

if __name__ == '__main__':
    setup_logging()
    print(answer("How much does the Growth plan cost per user per month?"))