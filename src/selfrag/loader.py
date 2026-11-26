import logging

from selfrag.config import settings
from selfrag.logging_setup import setup_logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def load_docs(path: str | Path | None = None) -> list[Document]:

    path = Path(path) if path else settings.data_dir
    pdf_paths = [path] if path.is_file() else sorted(path.glob('*.pdf'))

    docs=[]

    for pdf_path in pdf_paths:
        pages = PyPDFLoader(str(pdf_path)).load()
        logger.info("Loaded %s: %d pages", pdf_path.name, len(pages))
        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata['source'] = pdf_path.name
            page.metadata['page'] = page.metadata['page']+1
        docs.extend(pages)
    return docs


def chunk_docs(docs: list[Document], size: int | None = None, overlap: int | None = None) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = size if size is not None else settings.chunk_size,
        chunk_overlap = overlap if overlap is not None else settings.chunk_overlap,
        add_start_index = True
    )

    return splitter.split_documents(docs)


def clean_text(text: str) -> str:

    text = text.replace("\x7f\n", "- ")                  # bullet marker on its own line
    text = text.replace("\u25a0", "INR ")                # unmapped currency glyph
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text) # any leftover control characters
    text = re.sub(r"[ \t]+\n", "\n", text)               # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)               # long runs of blank lines
    return text.strip()


if __name__ == '__main__':
    setup_logging(logging.DEBUG)
    docs = load_docs()
    chunks = chunk_docs(docs)
    logger.info("%d pages -> %d chunks", len(docs), len(chunks))
    for c in chunks[:3]:
        logger.debug("%s len=%d\n%s", c.metadata, len(c.page_content), c.page_content)