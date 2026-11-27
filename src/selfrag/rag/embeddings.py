import logging
from functools import lru_cache

from selfrag.core.config import settings
from selfrag.core.logging import setup_logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dim)
    return embeddings

if __name__ == '__main__':
    vec = get_embeddings().embed_query('hello')
    setup_logging()
    logger.info("embedding length %d (expected %d)", len(vec), settings.embedding_dim)
