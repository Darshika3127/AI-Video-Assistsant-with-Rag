import threading
import time

import httpx
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_ollama import ChatOllama


_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)
_request_lock = threading.Lock()
_MAX_RETRIES = 3

set_llm_cache(InMemoryCache(maxsize=256))


class LLMRequestError(RuntimeError):
    """A local Ollama request failed after bounded retries."""


def _http_status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(
        response,
        "status_code",
        getattr(error, "status_code", None),
    )
    if isinstance(status_code, int):
        return status_code

    message = str(error)
    if "429" in message:
        return 429
    if "503" in message:
        return 503
    return None


def invoke_llm(runnable, input_value):
    """Run requests sequentially and retry local overload/service errors with backoff."""
    with _request_lock:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return runnable.invoke(input_value)
            except Exception as error:
                if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
                    raise LLMRequestError(
                        "Cannot connect to Ollama at http://localhost:11434. Start Ollama and make sure qwen3:8b is installed."
                    ) from error

                status_code = _http_status_code(error)
                if status_code not in (429, 503):
                    raise

                if attempt == _MAX_RETRIES:
                    raise LLMRequestError(
                        "Ollama remained unavailable (HTTP 503) or busy (HTTP 429) after "
                        f"{_MAX_RETRIES + 1} attempts. Check that Ollama is running at "
                        "http://localhost:11434 and that qwen3:8b is available."
                    ) from error

                time.sleep(2 ** (attempt + 1))


def get_llm(temperature: float = 0.3) -> ChatOllama:
    return ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=temperature,
        num_gpu=0,
        reasoning=False,
        rate_limiter=_rate_limiter,
        cache=True,
    )