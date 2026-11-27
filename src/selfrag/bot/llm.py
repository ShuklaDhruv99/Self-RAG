import logging
from functools import lru_cache
from typing import Any

from google.genai import errors as genai_errors
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from selfrag.core.config import settings

logger = logging.getLogger(__name__)

# 429 rate limit, 500 internal error, 503 overloaded, 504 deadline exceeded
RETRYABLE_CODES = {429, 500, 503, 504}


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        timeout=30,
        max_retries=2,
        temperature=0
    )


def is_transient(exc: BaseException) -> bool:
    
    return isinstance(exc, genai_errors.APIError) and exc.code in RETRYABLE_CODES


def invoke_with_retry(runnable, inputs: Any, attempts: int = 5):

    retryer = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception(is_transient),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    return retryer(runnable.invoke, inputs)
