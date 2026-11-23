"""Check that LLM calls work."""

from langchain_google_genai import ChatGoogleGenerativeAI
from selfrag.config import settings

def main():
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.google_api_key, timeout=20, max_retries=2)
    response = llm.invoke('Say hi in 5 words')

    print(response.content)

if __name__ == '__main__':
    main()