"""Main OpenAI LLM client wrapper"""

import asyncio
import inspect
import logging
from typing import Optional, AsyncIterator
from logorator import Logger
from .config import OpenAIConfig
from .responses_api import ResponsesAPI
from .chat_completions_api import ChatCompletionsAPI
from ..models import TextRequest, MessageRequest, TextResponse, StreamChunk
from ..utils import TwoLevelCache, retry_on_error

logger = logging.getLogger('smartllm')


class OpenAILLMClient:
    """Async client for text generation with OpenAI LLMs"""

    def __init__(self, config: Optional[OpenAIConfig] = None, max_concurrent: Optional[int] = None, dynamo_table_name: Optional[str] = None, cache_ttl_days: Optional[float] = None):
        """Initialize the OpenAI client
        
        Args:
            config: OpenAIConfig instance. If None, creates default config.
            max_concurrent: Max concurrent requests. Overrides config.max_concurrent if provided.
            dynamo_table_name: DynamoDB table name for shared cache. If None, only local cache is used.
            cache_ttl_days: TTL for DynamoDB cache entries in days. Defaults to 365.
        """
        self.config = config or OpenAIConfig()
        self.config.validate()
        self.client = None
        cache_kwargs = {"dynamo_table_name": dynamo_table_name}
        if cache_ttl_days is not None:
            cache_kwargs["ttl_days"] = cache_ttl_days
        self.cache = TwoLevelCache(**cache_kwargs)
        self._semaphore = None
        self._max_concurrent = max_concurrent if max_concurrent is not None else self.config.max_concurrent
        
        # API handlers (initialized after client)
        self.responses_api = None
        self.chat_completions_api = None

    async def _init_client(self):
        """Initialize OpenAI async client"""
        try:
            from openai import AsyncOpenAI
            import httpx

            # Match httpx connection pool to concurrency limit to avoid HTTP-layer bottleneck
            pool_size = self._max_concurrent or 100
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=pool_size,
                    max_keepalive_connections=pool_size,
                ),
            )

            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                max_retries=0,  # We handle retries ourselves
                http_client=http_client,
            )
            if self._max_concurrent:
                self._semaphore = asyncio.Semaphore(self._max_concurrent)
            
            # Initialize API handlers
            self.responses_api = ResponsesAPI(self.client, self.config, self.cache, self._semaphore)
            self.chat_completions_api = ChatCompletionsAPI(self.client, self.config, self.cache, self._semaphore)
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")
        except Exception:
            raise

    async def close(self):
        """Close the client connections"""
        if self.client:
            await self.client.close()

    async def list_available_models(self) -> list:
        """List all available OpenAI models"""
        if not self.client:
            await self._init_client()
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            return []

    def __str__(self):
        return f"OpenAILLMClient(default={self.config.default_model})"

    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _invoke_with_retry(self, func, on_progress=None, retry_context=None, **kwargs):
        """Invoke API with retry logic and context-aware logging
        
        Args:
            func: The async API function to call
            on_progress: Optional callback for progress events (from request)
            retry_context: Dict with model, max_tokens for logging context
            **kwargs: Arguments passed to the API function
        """
        ctx = retry_context or {}
        model = ctx.get("model", "unknown")
        max_tokens = ctx.get("max_tokens")

        @retry_on_error(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay,
            max_delay=self.config.max_retry_delay,
            on_retry=self._make_retry_handler(on_progress, model, max_tokens),
        )
        async def _invoke():
            return await func(**kwargs)
        
        return await _invoke()

    def _make_retry_handler(self, on_progress, model, max_tokens):
        """Create retry handler that logs context and fires on_progress callback"""
        async def _handle_retry(attempt, max_retries, error, delay):
            error_name = type(error).__name__
            error_msg = str(error)
            logger.warning(
                f"Retry {attempt}/{max_retries} after {error_name} on model {model} "
                f"(max_tokens={max_tokens}). Waiting {delay:.1f}s... Error: {error_msg}"
            )
            if on_progress is not None:
                event = {
                    "event": "retry",
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "error": error_name,
                    "error_message": error_msg,
                    "model": model,
                    "max_tokens": max_tokens,
                    "delay": round(delay, 1),
                }
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(event)
                else:
                    on_progress(event)

        return _handle_retry

    async def generate_text(self, request: TextRequest) -> TextResponse:
        """Generate text from a prompt
        
        Args:
            request: TextRequest with prompt and parameters
            
        Returns:
            TextResponse with generated text
        """
        if not self.client:
            await self._init_client()
        
        if request.api_type == "responses":
            return await self.responses_api.generate_text(request, self._invoke_with_retry)
        else:
            return await self.chat_completions_api.generate_text(request, self._invoke_with_retry)

    async def generate_text_stream(self, request: TextRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation
        
        Args:
            request: TextRequest with prompt and parameters
            
        Yields:
            StreamChunk objects with partial text
        """
        if not self.client:
            await self._init_client()
        
        # Only Chat Completions supports streaming for now
        if request.api_type == "responses":
            raise NotImplementedError("Streaming not yet supported for Response API")
        
        async for chunk in self.chat_completions_api.generate_text_stream(request):
            yield chunk

    async def send_message(self, request: MessageRequest) -> TextResponse:
        """Send a message in a conversation
        
        Args:
            request: MessageRequest with message history
            
        Returns:
            TextResponse with assistant's response
        """
        if not self.client:
            await self._init_client()
        
        # Only Chat Completions supports multi-turn for now
        if request.api_type == "responses":
            raise NotImplementedError("Multi-turn conversations not yet supported for Response API")
        
        return await self.chat_completions_api.send_message(request, self._invoke_with_retry)

    async def send_message_stream(self, request: MessageRequest) -> AsyncIterator[StreamChunk]:
        """Stream a conversation message
        
        Args:
            request: MessageRequest with message history
            
        Yields:
            StreamChunk objects with partial responses
        """
        if not self.client:
            await self._init_client()
        
        # Only Chat Completions supports streaming for now
        if request.api_type == "responses":
            raise NotImplementedError("Streaming not yet supported for Response API")
        
        async for chunk in self.chat_completions_api.send_message_stream(request):
            yield chunk
