from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    gemini_api_key: str
    gemini_model: str
    database_url: str

    # Embeddings
    gemini_embedding_model: str
    embedding_dim: int = 768

    # Vector store
    collection_name: str = "docs"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4

    # Self-RAG loop limits
    max_rewrites: int = 2
    max_generations: int = 2

    # Files
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def langchain_database_url(self) -> str:
        """URL in the form LangChain's PGVector expects (psycopg v3 driver)."""
        return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)


settings = Settings()
