import logging
import time

from selfrag.loader import chunk_docs, load_docs
from selfrag.logging_setup import setup_logging
from selfrag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def ingest(reset: bool = True) -> int:
    docs = load_docs()
    chunks = chunk_docs(docs)
    logger.info("%d pages -> %d chunks", len(docs), len(chunks))

    start = time.perf_counter()
    store = get_vectorstore(reset=reset)
    store.add_documents(chunks)
    logger.info("Stored %d chunks (reset=%s) in %.1fs", len(chunks), reset, time.perf_counter() - start)
    return len(chunks)


if __name__ == "__main__":
    setup_logging()
    ingest()