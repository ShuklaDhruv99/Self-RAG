from selfrag.config import settings
from selfrag.vectorstore import get_vectorstore

def get_retriever(k: int | None = None):
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={'k': k if k is not None else settings.top_k})
    return retriever

if __name__ == "__main__":
    questions = [
        "How much does the Growth plan cost per user per month?",
        "Which plan includes NexaSupport?",
        "How many queries per month does Starter allow?",
        "What discount does annual billing give?",
        "Does NexaChat support role-based access control?",
        "Does NexaChat integrate with Telegram?",
    ]

    retriever = get_retriever()
    store = get_vectorstore()

    for question in questions:
        print("\nQ:", question)
        for rank, doc in enumerate(retriever.invoke(question), start=1):
            print(" ", rank, doc.metadata["page"], doc.page_content[:80].replace("\n", " "))

        print("  scores (lower = closer):")
        for doc, score in store.similarity_search_with_score(question, k=4):
            print(f"    {score:.3f}  p{doc.metadata['page']}")