"""Main OpenAI LLM client wrapper"""

import asyncio
import logging
from typing import Optional, AsyncIterator
from .config import OpenAIConfig
from .responses_api import ResponsesAPI
from .chat_completions_api import ChatCompletionsAPI
from ..models import TextRequest, MessageRequest, TextResponse, StreamChunk
from ..utils import JSONFileCache, setup_logging, retry_on_error

logger = setup_logging()


class OpenAILLMClient:
    """Async client for text generation with OpenAI LLMs"""

    def __init__(self, config: Optional[OpenAIConfig] = None, max_concurrent: Optional[int] = None):
        """Initialize the OpenAI client
        
        Args:
            config: OpenAIConfig instance. If None, creates default config.
            max_concurrent: Max concurrent requests. Overrides config.max_concurrent if provided.
        """
        self.config = config or OpenAIConfig()
        self.config.validate()
        self.client = None
        self.cache = JSONFileCache()
        self._semaphore = None
        self._max_concurrent = max_concurrent if max_concurrent is not None else self.config.max_concurrent
        
        # API handlers (initialized after client)
        self.responses_api = None
        self.chat_completions_api = None

    async def _init_client(self):
        """Initialize OpenAI async client"""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                max_retries=0,  # We handle retries ourselves
            )
            if self._max_concurrent:
                self._semaphore = asyncio.Semaphore(self._max_concurrent)
            
            # Initialize API handlers
            self.responses_api = ResponsesAPI(self.client, self.config, self.cache, self._semaphore)
            self.chat_completions_api = ChatCompletionsAPI(self.client, self.config, self.cache, self._semaphore)
            
            logger.debug(f"OpenAI client initialized")
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
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
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _invoke_with_retry(self, func, **kwargs):
        """Invoke API with retry logic"""
        @retry_on_error(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay,
            max_delay=self.config.max_retry_delay,
        )
        async def _invoke():
            return await func(**kwargs)
        
        return await _invoke()

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
