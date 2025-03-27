import asyncio
from collections.abc import Callable
from hashlib import sha256

from cacherator import Cached, JSONCache
from logorator import Logger
from openai import APIConnectionError, APIError, RateLimitError

from .async_providers import AsyncOpenAIProvider


async def _with_retries(coro: Callable, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await coro()
        except (RateLimitError, APIConnectionError, APIError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                delay = base_delay
                Logger.note(f"[Retry {attempt + 1}] Error: {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                raise  # Re-raise last error if out of retries


def _default_callback(chunk: str):
    Logger.note(chunk, mode="short")


class AsyncLLM(JSONCache):
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_P = 0.9
    DEFAULT_FREQUENCY_PENALTY = 1.0
    DEFAULT_PRESENCE_PENALTY = 0.0
    DEFAULT_MAX_TOKENS = 10_000

    def __init__(
            self, clear_cache: bool = False, ttl: int = 7, **kwargs):
        self.prompt = kwargs.get('prompt')
        self.temperature = kwargs.get('temperature', self.DEFAULT_TEMPERATURE)
        self.top_p = kwargs.get('top_p', self.DEFAULT_TOP_P)
        self.frequency_penalty = kwargs.get('frequency_penalty', self.DEFAULT_FREQUENCY_PENALTY)
        self.presence_penalty = kwargs.get('presence_penalty', self.DEFAULT_PRESENCE_PENALTY)
        self.system_prompt = kwargs.get('system_prompt')
        self.search_recency_filter = kwargs.get('search_recency_filter')
        self.return_citations = kwargs.get('return_citations', False)
        self.json_schema = kwargs.get('json_schema')
        self.clear_cache = clear_cache
        self.ttl = ttl
        self.model = kwargs.get('model')
        self.base = kwargs.get('base')
        self.api_key = kwargs.get('api_key')
        self.max_input_tokens = kwargs.get('max_input_tokens', self.DEFAULT_MAX_TOKENS)
        self.max_output_tokens = kwargs.get('max_output_tokens', self.DEFAULT_MAX_TOKENS)
        self.stream = kwargs.get('stream', False)
        self.callback = kwargs.get('callback', _default_callback)
        self.max_retries = kwargs.get("max_retries", 20)
        self.base_delay = kwargs.get("base_delay", 15)
        self.kwargs = kwargs

        self.response: str | dict | None = None
        self.usage: dict | None = None

        super().__init__(data_id=self.identifier, directory="data/llm", ttl=ttl, clear_cache=clear_cache)
        self._excluded_cache_vars = ["api_key", "_hash_information", "kwargs"]

    def __str__(self):
        return self.readable_identifier

    def __repr__(self):
        return self.__str__()

    @property
    def identifier(self) -> str:
        """Generate a unique identifier for this configuration for caching purposes"""
        prompt_str = str(self.prompt)
        truncated_prompt = prompt_str[:30] + "..." if len(prompt_str) > 30 else prompt_str
        base_id = f"{self.base}_{self.model}_{truncated_prompt}"
        hash_input = str(self.kwargs)
        _hash = sha256(hash_input.encode()).hexdigest()[:10]
        return f"{base_id}_{_hash}"

    @property
    def readable_identifier(self) -> str:
        return f"{self.base}  ({self.prompt[:60]}){' (JSON)' if self.json_schema else ''}"

    @Cached()
    def provider(self):
        if self.base == "openai":
            return AsyncOpenAIProvider(**self.__dict__)
        else:
            raise ValueError(f"Base Model '{self.base}' not supported.")

    @Logger()
    async def execute(self):
        if self.response is None:
            self.response = await _with_retries(lambda: self.provider().execute(), max_retries=self.max_retries, base_delay=self.base_delay)
            self.usage = self.provider().usage()
        return self.response

    @Logger()
    async def models(self):
        return await self.provider().models()

    def usage(self):
        return self.usage
