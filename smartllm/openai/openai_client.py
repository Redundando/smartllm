"""Main OpenAI LLM client wrapper"""

import json
import logging
import time
import asyncio
from typing import Optional, AsyncIterator, Dict, Any, Type
from pydantic import BaseModel
from .config import OpenAIConfig
from ..models import TextRequest, MessageRequest, TextResponse, StreamChunk
from ..utils import pydantic_to_tool_schema, JSONFileCache, setup_logging, retry_on_error

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
            
        model = request.model or self.config.default_model
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key
        cache_key = None
        if temperature == 0 and not request.stream:
            cache_key = self._generate_cache_key(
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None
            )
        
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
            logger.info(f"Cleared cache entry: {cache_key[:8]}...")
        
        if request.use_cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit [{cache_key[:8]}] - {model} - prompt: {request.prompt[:50]}...")
                return self._deserialize_response(cached["data"], request.response_format)
        
        prompt_preview = request.prompt[:60] + "..." if len(request.prompt) > 60 else request.prompt
        logger.info(f"API call to {model} - temp={temperature} - prompt: {prompt_preview}")
        
        start_time = time.time()
        
        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        # Build request params
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "top_p": request.top_p or self.config.top_p,
        }
        
        # Add structured output if requested
        if request.response_format:
            params["response_format"] = {"type": "json_object"}
            params["tools"] = [self._build_tool_schema(request.response_format)]
            params["tool_choice"] = {"type": "function", "function": {"name": params["tools"][0]["function"]["name"]}}

        try:
            if self._semaphore:
                async with self._semaphore:
                    response = await self._invoke_with_retry(self.client.chat.completions.create, **params)
            else:
                response = await self._invoke_with_retry(self.client.chat.completions.create, **params)
            
            result = self._parse_response(response, model, request.response_format)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Response received - {result.input_tokens} in / {result.output_tokens} out tokens - "
                f"{elapsed:.2f}s - {result.text[:50]}..."
            )
            
            if cache_key:
                cache_metadata = {
                    "prompt": request.prompt,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": request.max_tokens or self.config.max_tokens,
                    "system_prompt": request.system_prompt,
                    "response_format": request.response_format.__name__ if request.response_format else None,
                }
                self.cache.set(cache_key, self._serialize_response(result), cache_metadata)
                logger.debug(f"Cached response: {cache_key[:8]}...")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error after {elapsed:.2f}s - {model}: {str(e)}")
            raise

    async def generate_text_stream(self, request: TextRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation
        
        Args:
            request: TextRequest with prompt and parameters
            
        Yields:
            StreamChunk objects with partial text
        """
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature or self.config.temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "stream": True,
        }

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamChunk(text=chunk.choices[0].delta.content, model=model)
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise

    async def send_message(self, request: MessageRequest) -> TextResponse:
        """Send a message in a conversation
        
        Args:
            request: MessageRequest with message history
            
        Returns:
            TextResponse with assistant's response
        """
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key
        cache_key = None
        if temperature == 0 and not request.stream:
            messages_str = json.dumps([{"role": m.role, "content": m.content} for m in request.messages])
            cache_key = self._generate_cache_key(
                model=model,
                messages=messages_str,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None
            )
        
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
            logger.info(f"Cleared cache entry: {cache_key[:8]}...")
        
        if request.use_cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit [{cache_key[:8]}] - {model} - {len(request.messages)} messages")
                return self._deserialize_response(cached["data"], request.response_format)
        
        last_msg = request.messages[-1].content[:60] if request.messages else ""
        logger.info(f"API call to {model} - temp={temperature} - {len(request.messages)} messages - last: {last_msg}...")
        
        start_time = time.time()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
        }
        
        if request.response_format:
            params["response_format"] = {"type": "json_object"}
            params["tools"] = [self._build_tool_schema(request.response_format)]
            params["tool_choice"] = {"type": "function", "function": {"name": params["tools"][0]["function"]["name"]}}

        try:
            if self._semaphore:
                async with self._semaphore:
                    response = await self._invoke_with_retry(self.client.chat.completions.create, **params)
            else:
                response = await self._invoke_with_retry(self.client.chat.completions.create, **params)
            
            result = self._parse_response(response, model, request.response_format)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Response received - {result.input_tokens} in / {result.output_tokens} out tokens - "
                f"{elapsed:.2f}s - {result.text[:50]}..."
            )
            
            if cache_key:
                cache_metadata = {
                    "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": request.max_tokens or self.config.max_tokens,
                    "system_prompt": request.system_prompt,
                    "response_format": request.response_format.__name__ if request.response_format else None,
                }
                self.cache.set(cache_key, self._serialize_response(result), cache_metadata)
                logger.debug(f"Cached response: {cache_key[:8]}...")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error after {elapsed:.2f}s - {model}: {str(e)}")
            raise

    async def send_message_stream(self, request: MessageRequest) -> AsyncIterator[StreamChunk]:
        """Stream a conversation message
        
        Args:
            request: MessageRequest with message history
            
        Yields:
            StreamChunk objects with partial responses
        """
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature or self.config.temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "stream": True,
        }

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamChunk(text=chunk.choices[0].delta.content, model=model)
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise

    def _build_tool_schema(self, response_format: Type[BaseModel]) -> Dict[str, Any]:
        """Build OpenAI tool schema from Pydantic model"""
        schema = pydantic_to_tool_schema(response_format)
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"]
            }
        }

    def _parse_response(self, response, model: str, response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Parse OpenAI response"""
        choice = response.choices[0]
        
        # Check for tool calls (structured output)
        if choice.message.tool_calls and response_format:
            tool_call = choice.message.tool_calls[0]
            tool_input = json.loads(tool_call.function.arguments)
            structured_data = response_format(**tool_input)
            text = json.dumps(tool_input, indent=2)
        else:
            text = choice.message.content or ""
            structured_data = None
        
        return TextResponse(
            text=text,
            model=model,
            stop_reason=choice.finish_reason or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            structured_data=structured_data,
        )

    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key from request parameters"""
        return self.cache._generate_key(**kwargs)
    
    def _serialize_response(self, response: TextResponse) -> Dict[str, Any]:
        """Serialize TextResponse for caching"""
        return {
            "text": response.text,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "metadata": response.metadata,
            "structured_data": response.structured_data.model_dump() if response.structured_data else None,
        }
    
    def _deserialize_response(self, data: Dict[str, Any], response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Deserialize cached data back to TextResponse"""
        structured_data = None
        if data.get("structured_data") and response_format:
            structured_data = response_format(**data["structured_data"])
        
        return TextResponse(
            text=data["text"],
            model=data["model"],
            stop_reason=data["stop_reason"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            metadata=data.get("metadata", {}),
            structured_data=structured_data,
        )
