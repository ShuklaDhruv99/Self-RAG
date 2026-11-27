import logging

from langchain_postgres import PGVector
from selfrag.core.config import settings
from selfrag.rag.embeddings import get_embeddings
from selfrag.core.logging import setup_logging

logger = logging.getLogger(__name__)

def get_vectorstore(reset: bool = False) -> PGVector:

    logger.debug("Opening collection %r (reset=%s)", settings.collection_name, reset)
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=settings.langchain_database_url,
        use_jsonb=True,
        pre_delete_collection=reset)


if __name__ == "__main__":
    store = get_vectorstore()
    setup_logging()
    logger.info("vector store ready: %s", type(store).__name__)