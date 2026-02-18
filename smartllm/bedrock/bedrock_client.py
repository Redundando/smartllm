"""Main Bedrock LLM client wrapper"""

import json
import logging
import time
import asyncio
from typing import Optional, AsyncIterator, List, Dict, Any, Type
from pydantic import BaseModel
from .config import BedrockConfig
from ..models import (
    TextRequest, 
    MessageRequest,
    TextResponse, 
    StreamChunk,
)
from ..utils import pydantic_to_tool_schema, JSONFileCache, setup_logging, retry_on_error

logger = setup_logging()

# Default Bedrock model quotas for concurrency limiting
DEFAULT_MODEL_QUOTAS = {
    'claude-3-5-sonnet-v2': {'rpm': 10, 'tpm': 200000, 'concurrent': 1},
    'claude-3-5-sonnet': {'rpm': 200, 'tpm': 400000, 'concurrent': 2},
    'claude-3-sonnet': {'rpm': 200, 'tpm': 400000, 'concurrent': 2},
    'claude-3-haiku': {'rpm': 400, 'tpm': 400000, 'concurrent': 5},
    'claude-3-opus': {'rpm': 50, 'tpm': 200000, 'concurrent': 1},
    'llama': {'rpm': 500, 'tpm': 500000, 'concurrent': 5},
    'mistral': {'rpm': 300, 'tpm': 300000, 'concurrent': 3},
    'titan': {'rpm': 400, 'tpm': 400000, 'concurrent': 5},
}


class BedrockLLMClient:
    """Async client for text generation with AWS Bedrock LLMs"""

    def __init__(self, config: Optional[BedrockConfig] = None, max_concurrent: Optional[int] = None):
        """Initialize the Bedrock client
        
        Args:
            config: BedrockConfig instance. If None, creates default config.
            max_concurrent: Max concurrent requests. Overrides config.max_concurrent if provided.
        """
        self.config = config or BedrockConfig()
        self.config.validate()
        self.client = None
        self.models_client = None
        self.cache = JSONFileCache()
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._max_concurrent = max_concurrent if max_concurrent is not None else self.config.max_concurrent

    async def _init_client(self):
        """Initialize aioboto3 Bedrock client"""
        try:
            import aioboto3
            creds = self.config.get_credentials()
            session = aioboto3.Session()
            self.client = await session.client("bedrock-runtime", **creds).__aenter__()
            self.models_client = await session.client("bedrock", **creds).__aenter__()
            logger.debug(f"Bedrock client initialized - region: {creds['region_name']}")
        except ImportError:
            raise ImportError("aioboto3 is required. Install with: pip install aioboto3")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    async def close(self):
        """Close the client connections"""
        if self.client:
            await self.client.__aexit__(None, None, None)
        if self.models_client:
            await self.models_client.__aexit__(None, None, None)

    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def _get_semaphore(self, model: str) -> asyncio.Semaphore:
        """Get or create semaphore for model to limit concurrent requests"""
        if model not in self._semaphores:
            # Use explicit max_concurrent or infer from model defaults
            if self._max_concurrent:
                limit = self._max_concurrent
            else:
                # Get default concurrent limit for this model
                limit = 2  # Safe default
                for pattern, quotas in DEFAULT_MODEL_QUOTAS.items():
                    if pattern in model.lower():
                        limit = quotas['concurrent']
                        break
            
            self._semaphores[model] = asyncio.Semaphore(limit)
            logger.debug(f"Created semaphore for {model} with limit={limit}")
        
        return self._semaphores[model]

    async def _invoke_model_with_retry(self, **kwargs):
        """Invoke model with retry logic"""
        @retry_on_error(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay,
            max_delay=self.config.max_retry_delay,
        )
        async def _invoke():
            return await self.client.invoke_model(**kwargs)
        
        return await _invoke()

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in Bedrock"""
        if not self.models_client:
            await self._init_client()
        try:
            response = await self.models_client.list_foundation_models()
            return response.get("modelSummaries", [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def list_available_model_ids(self) -> List[str]:
        """List all available model IDs in Bedrock
        
        Returns:
            List of model ID strings
        """
        models = await self.list_available_models()
        return [m.get("modelId") for m in models if isinstance(m, dict) and m.get("modelId")]

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
        # If no temperature specified, use 0 (deterministic + cacheable)
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key for this specific request
        cache_key = None
        if temperature == 0 and not request.stream:
            cache_key = self._generate_cache_key(
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None
            )
        
        # Clear this specific cache entry if requested
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
            logger.info(f"Cleared cache entry: {cache_key[:8]}...")
        
        # Check cache only if caching enabled
        if request.use_cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit [{cache_key[:8]}] - {model} - prompt: {request.prompt[:50]}...")
                return self._deserialize_response(cached["data"], request.response_format)
        
        # Log API call
        prompt_preview = request.prompt[:60] + "..." if len(request.prompt) > 60 else request.prompt
        logger.info(f"API call to {model} - temp={temperature} - prompt: {prompt_preview}")
        
        start_time = time.time()
        
        body = self._build_request_body(
            model=model,
            prompt=request.prompt,
            temperature=temperature,
            max_tokens=request.max_tokens or self.config.max_tokens,
            top_p=request.top_p or self.config.top_p,
            top_k=request.top_k or self.config.top_k,
            system_prompt=request.system_prompt,
            response_format=request.response_format,
        )

        try:
            semaphore = self._get_semaphore(model)
            async with semaphore:
                response = await self._invoke_model_with_retry(
                    modelId=model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
            
            response_body = json.loads(await response["body"].read())
            result = self._parse_response(response_body, model, request.response_format)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Response received - {result.input_tokens} in / {result.output_tokens} out tokens - "
                f"{elapsed:.2f}s - {result.text[:50]}..."
            )
            
            # Cache if applicable
            if cache_key:
                cache_metadata = {
                    "prompt": request.prompt,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": request.max_tokens or self.config.max_tokens,
                    "system_prompt": request.system_prompt,
                    "response_format": request.response_format.__name__ if request.response_format else None,
                    "top_p": request.top_p or self.config.top_p,
                    "top_k": request.top_k or self.config.top_k,
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
        
        body = self._build_request_body(
            model=model,
            prompt=request.prompt,
            temperature=request.temperature or self.config.temperature,
            max_tokens=request.max_tokens or self.config.max_tokens,
            top_p=request.top_p or self.config.top_p,
            top_k=request.top_k or self.config.top_k,
        )

        try:
            response = await self.client.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
            )
            
            async for event in response["body"]:
                if "chunk" in event:
                    chunk_data = json.loads(event["chunk"]["bytes"])
                    text = self._extract_text_from_chunk(chunk_data, model)
                    if text:
                        yield StreamChunk(text=text, model=model)
                        
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
        # If no temperature specified, use 0 (deterministic + cacheable)
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key for this specific request
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
        
        # Clear this specific cache entry if requested
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
            logger.info(f"Cleared cache entry: {cache_key[:8]}...")
        
        # Check cache only if caching enabled
        if request.use_cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit [{cache_key[:8]}] - {model} - {len(request.messages)} messages")
                return self._deserialize_response(cached["data"], request.response_format)
        
        # Log API call
        last_msg = request.messages[-1].content[:60] if request.messages else ""
        logger.info(f"API call to {model} - temp={temperature} - {len(request.messages)} messages - last: {last_msg}...")
        
        start_time = time.time()
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": temperature,
        }
        
        if request.system_prompt:
            body["system"] = request.system_prompt
            
        if request.response_format and "claude" in model.lower():
            tool_schema = pydantic_to_tool_schema(request.response_format)
            body["tools"] = [tool_schema]
            body["tool_choice"] = {"type": "tool", "name": tool_schema["name"]}

        try:
            semaphore = self._get_semaphore(model)
            async with semaphore:
                response = await self._invoke_model_with_retry(
                    modelId=model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
            
            response_body = json.loads(await response["body"].read())
            result = self._parse_response(response_body, model, request.response_format)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Response received - {result.input_tokens} in / {result.output_tokens} out tokens - "
                f"{elapsed:.2f}s - {result.text[:50]}..."
            )
            
            # Cache if applicable
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
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
        }
        
        if request.system_prompt:
            body["system"] = request.system_prompt

        try:
            response = await self.client.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
            )
            
            async for event in response["body"]:
                if "chunk" in event:
                    chunk_data = json.loads(event["chunk"]["bytes"])
                    text = self._extract_text_from_chunk(chunk_data, model)
                    if text:
                        yield StreamChunk(text=text, model=model)
                        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise

    def _build_request_body(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        system_prompt: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Build request body for text generation based on model type"""
        
        if "claude" in model.lower():
            # Claude 3+ models use Messages API
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                body["system"] = system_prompt
            if response_format:
                tool_schema = pydantic_to_tool_schema(response_format)
                body["tools"] = [tool_schema]
                body["tool_choice"] = {"type": "tool", "name": tool_schema["name"]}
        elif "llama" in model.lower():
            # Llama models
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        elif "mistral" in model.lower():
            # Mistral models
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            # Default/generic format
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        
        return body

    def _parse_response(self, response_body: Dict[str, Any], model: str, response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Parse response based on model type"""
        if "claude" in model.lower():
            # Check for tool use (structured output)
            content = response_body.get("content", [])
            if content and content[0].get("type") == "tool_use" and response_format:
                tool_input = content[0].get("input", {})
                structured_data = response_format(**tool_input)
                text = json.dumps(tool_input, indent=2)
            else:
                # Regular text response
                text = response_body["content"][0]["text"]
                structured_data = None
                
            stop_reason = response_body.get("stop_reason", "")
            input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
            output_tokens = response_body.get("usage", {}).get("output_tokens", 0)
        elif "llama" in model.lower():
            # Llama response format
            text = response_body.get("generation", "")
            stop_reason = response_body.get("stop_reason", "")
            input_tokens = 0
            output_tokens = 0
        elif "mistral" in model.lower():
            # Mistral response format
            outputs = response_body.get("outputs", [])
            text = outputs[0].get("text", "") if outputs else ""
            stop_reason = outputs[0].get("stop_reason", "") if outputs else ""
            input_tokens = 0
            output_tokens = 0
        else:
            # Generic handling
            text = response_body.get("generated_text", response_body.get("generation", ""))
            stop_reason = response_body.get("stop_reason", "")
            input_tokens = 0
            output_tokens = 0

        return TextResponse(
            text=text,
            model=model,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            structured_data=structured_data,
        )

    def _extract_text_from_chunk(self, chunk_data: Dict[str, Any], model: str) -> str:
        """Extract text from streaming chunk based on model type"""
        if "claude" in model.lower():
            if "content_block_start" in chunk_data:
                return ""
            if "content_block_delta" in chunk_data:
                return chunk_data["content_block_delta"]["delta"].get("text", "")
        elif "llama" in model.lower():
            return chunk_data.get("generation", "")
        
        return ""
    
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
