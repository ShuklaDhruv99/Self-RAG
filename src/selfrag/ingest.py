import logging

from selfrag.loader import chunk_docs, load_docs
from selfrag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def ingest(reset: bool = True) -> int:
    docs = load_docs()
    chunks = chunk_docs(docs)
    logger.info("%d pages -> %d chunks", len(docs), len(chunks))

    store = get_vectorstore(reset=reset)
    store.add_documents(chunks)
    logger.info("Stored %d chunks (reset=%s)", len(chunks), reset)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ingest()