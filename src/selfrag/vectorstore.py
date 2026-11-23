from langchain_postgres import PGVector
from selfrag.config import settings
from selfrag.embeddings import get_embeddings

def get_vectorstore(reset: bool = False) -> PGVector:

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=settings.langchain_database_url,
        use_jsonb=True,
        pre_delete_collection=reset)


if __name__ == "__main__":
    store = get_vectorstore()
    print(type(store))