"""Main Bedrock LLM client wrapper"""

import json
import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, AsyncIterator, List, Dict, Any, Type
from pydantic import BaseModel
from logorator import Logger
from .config import BedrockConfig
from ..models import (
    TextRequest, 
    MessageRequest,
    TextResponse, 
    StreamChunk,
)
from ..utils import pydantic_to_tool_schema, TwoLevelCache, retry_on_error
from ..defaults import BEDROCK_THINKING_BUDGET

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

    def __init__(self, config: Optional[BedrockConfig] = None, max_concurrent: Optional[int] = None, dynamo_table_name: Optional[str] = None, cache_ttl_days: Optional[float] = None):
        """Initialize the Bedrock client
        
        Args:
            config: BedrockConfig instance. If None, creates default config.
            max_concurrent: Max concurrent requests. Overrides config.max_concurrent if provided.
            dynamo_table_name: DynamoDB table name for shared cache. If None, only local cache is used.
            cache_ttl_days: TTL for DynamoDB cache entries in days. Defaults to 365.
        """
        self.config = config or BedrockConfig()
        self.config.validate()
        self.client = None
        self.models_client = None
        cache_kwargs = {"dynamo_table_name": dynamo_table_name}
        if cache_ttl_days is not None:
            cache_kwargs["ttl_days"] = cache_ttl_days
        self.cache = TwoLevelCache(**cache_kwargs)
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._max_concurrent = max_concurrent if max_concurrent is not None else self.config.max_concurrent

    async def _init_client(self):
        """Initialize aioboto3 Bedrock client"""
        try:
            import aioboto3
            from botocore.config import Config

            creds = self.config.get_credentials()
            # Match connection pool size to concurrency limit to avoid HTTP-layer bottleneck
            pool_size = self._max_concurrent or 10
            boto_config = Config(max_pool_connections=pool_size)

            session = aioboto3.Session()
            self.client = await session.client(
                "bedrock-runtime", config=boto_config, **creds
            ).__aenter__()
            self.models_client = await session.client(
                "bedrock", config=boto_config, **creds
            ).__aenter__()
        except ImportError:
            raise ImportError("aioboto3 is required. Install with: pip install aioboto3")
        except Exception:
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
        except Exception:
            return []
    
    async def list_available_model_ids(self) -> List[str]:
        """List all available model IDs in Bedrock
        
        Returns:
            List of model ID strings
        """
        models = await self.list_available_models()
        return [m.get("modelId") for m in models if isinstance(m, dict) and m.get("modelId")]

    def __str__(self):
        return f"BedrockLLMClient(default={self.config.default_model})"

    def _resolve_thinking_budget(self, request: TextRequest) -> Optional[int]:
        """Resolve the thinking budget from request parameters.
        
        Returns budget_tokens if thinking is requested, None otherwise.
        Priority: budget_tokens > reasoning_effort mapping.
        """
        if request.budget_tokens:
            return max(request.budget_tokens, 1024)  # Minimum 1024 per API requirement
        if request.reasoning_effort:
            budget = BEDROCK_THINKING_BUDGET.get(request.reasoning_effort)
            if budget is None:
                raise ValueError(
                    f"Invalid reasoning_effort '{request.reasoning_effort}'. "
                    f"Must be one of: {', '.join(BEDROCK_THINKING_BUDGET.keys())}"
                )
            return budget
        return None

    @Logger(exclude_args=[])
    async def generate_text(self, request: TextRequest) -> TextResponse:
        """Generate text from a prompt"""
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        thinking_budget = self._resolve_thinking_budget(request)
        # If no temperature specified, use 0 (deterministic + cacheable)
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key for this specific request
        cache_key = None
        if (temperature == 0 or thinking_budget) and not request.stream:
            cache_key = self._generate_cache_key(
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None,
                reasoning_effort=request.reasoning_effort,
                budget_tokens=thinking_budget,
            )
        
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
        
        if request.use_cache and cache_key:
            cached, cache_source = self.cache.get(cache_key)
            if cached:
                Logger.note(f"Cache hit [{cache_key[:8]}] - {model}")
                result = self._deserialize_response(cached["data"], request.response_format, cached.get("metadata", {}))
                result.cache_source = cache_source
                result.cache_key = cache_key
                return result

        prompt_preview = request.prompt[:60] + "..." if len(request.prompt) > 60 else request.prompt
        Logger.note(f"{model} | temp={temperature} | thinking={thinking_budget or 'off'} | {prompt_preview}")
        
        # Two-pass approach when both thinking and structured output are requested
        if thinking_budget and request.response_format:
            result = await self._generate_with_thinking_and_structure(
                request=request,
                model=model,
                temperature=temperature,
                thinking_budget=thinking_budget,
            )
        elif thinking_budget:
            result = await self._generate_with_thinking(
                request=request,
                model=model,
                thinking_budget=thinking_budget,
            )
        else:
            result = await self._generate_standard(
                request=request,
                model=model,
                temperature=temperature,
            )
        
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
                "reasoning_effort": request.reasoning_effort,
                "budget_tokens": thinking_budget,
            }
            self.cache.set(cache_key, self._serialize_response(result), cache_metadata)
        result.cache_key = cache_key
        return result

    async def _generate_standard(self, request: TextRequest, model: str, temperature: float) -> TextResponse:
        """Standard generation without extended thinking."""
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

        semaphore = self._get_semaphore(model)
        async with semaphore:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response = await self._invoke_model_with_retry(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
            )
            elapsed = round(time.monotonic() - t0, 3)
        
        response_body = json.loads(await response["body"].read())
        result = self._parse_response(response_body, model, request.response_format)
        result.timestamp = started_at
        result.elapsed_seconds = elapsed
        result.metadata["prompt"] = request.prompt
        result.metadata["response_format"] = request.response_format.model_json_schema() if request.response_format else None
        Logger.note(f"{result.input_tokens} in / {result.output_tokens} out | {result.text[:50]}")
        return result

    async def _generate_with_thinking(self, request: TextRequest, model: str, thinking_budget: int) -> TextResponse:
        """Generation with extended thinking, no structured output."""
        max_tokens = request.max_tokens or self.config.max_tokens
        # budget_tokens must be less than max_tokens
        if thinking_budget >= max_tokens:
            max_tokens = thinking_budget + 4096

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        }
        if request.system_prompt:
            body["system"] = request.system_prompt

        semaphore = self._get_semaphore(model)
        async with semaphore:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response = await self._invoke_model_with_retry(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
            )
            elapsed = round(time.monotonic() - t0, 3)

        response_body = json.loads(await response["body"].read())
        result = self._parse_thinking_response(response_body, model)
        result.timestamp = started_at
        result.elapsed_seconds = elapsed
        result.metadata["prompt"] = request.prompt
        Logger.note(f"{result.input_tokens} in / {result.output_tokens} out (thinking={result.reasoning_tokens}) | {result.text[:50]}")
        return result

    async def _generate_with_thinking_and_structure(
        self, request: TextRequest, model: str, temperature: float, thinking_budget: int
    ) -> TextResponse:
        """Two-pass: thinking first, then structured extraction.
        
        Pass 1: Extended thinking to reason through the prompt.
        Pass 2: Forced tool use to extract structured output from pass 1 result.
        """
        # --- Pass 1: Think ---
        max_tokens = request.max_tokens or self.config.max_tokens
        if thinking_budget >= max_tokens:
            max_tokens = thinking_budget + 4096

        body_pass1 = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        }
        if request.system_prompt:
            body_pass1["system"] = request.system_prompt

        semaphore = self._get_semaphore(model)
        async with semaphore:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response1 = await self._invoke_model_with_retry(
                modelId=model,
                body=json.dumps(body_pass1),
                contentType="application/json",
            )

            response_body1 = json.loads(await response1["body"].read())
            pass1_result = self._parse_thinking_response(response_body1, model)
            Logger.note(f"Pass 1 (thinking): {pass1_result.input_tokens} in / {pass1_result.output_tokens} out (thinking={pass1_result.reasoning_tokens})")

            # --- Pass 2: Structure extraction ---
            tool_schema = pydantic_to_tool_schema(request.response_format)
            extraction_prompt = (
                "Extract the content from the following text into the required structured format. "
                "Map the information to the schema fields as accurately as possible. "
                "Do not add, invent, or omit any information — use only what is provided.\n\n"
                f"---\n{pass1_result.text}\n---"
            )

            body_pass2 = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": extraction_prompt}],
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": 0,
                "tools": [tool_schema],
                "tool_choice": {"type": "tool", "name": tool_schema["name"]},
            }

            response2 = await self._invoke_model_with_retry(
                modelId=model,
                body=json.dumps(body_pass2),
                contentType="application/json",
            )
            elapsed = round(time.monotonic() - t0, 3)

        response_body2 = json.loads(await response2["body"].read())
        pass2_result = self._parse_response(response_body2, model, request.response_format)
        Logger.note(f"Pass 2 (structure): {pass2_result.input_tokens} in / {pass2_result.output_tokens} out")

        # Combine results
        total_input = pass1_result.input_tokens + pass2_result.input_tokens
        total_output = pass1_result.output_tokens + pass2_result.output_tokens

        return TextResponse(
            text=pass1_result.text,
            model=model,
            stop_reason=pass2_result.stop_reason,
            input_tokens=total_input,
            output_tokens=total_output,
            reasoning_tokens=pass1_result.reasoning_tokens,
            timestamp=started_at,
            elapsed_seconds=elapsed,
            metadata={
                "prompt": request.prompt,
                "response_format": request.response_format.model_json_schema(),
                "pass1_tokens": {"input": pass1_result.input_tokens, "output": pass1_result.output_tokens},
                "pass2_tokens": {"input": pass2_result.input_tokens, "output": pass2_result.output_tokens},
            },
            structured_data=pass2_result.structured_data,
        )

    async def generate_text_stream(self, request: TextRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation
        
        Args:
            request: TextRequest with prompt and parameters
            
        Yields:
            StreamChunk objects with partial text. Thinking chunks have
            metadata["type"] = "thinking".
        """
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        thinking_budget = self._resolve_thinking_budget(request)

        if thinking_budget:
            max_tokens = request.max_tokens or self.config.max_tokens
            if thinking_budget >= max_tokens:
                max_tokens = thinking_budget + 4096
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": max_tokens,
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
            }
            if request.system_prompt:
                body["system"] = request.system_prompt
        else:
            body = self._build_request_body(
                model=model,
                prompt=request.prompt,
                temperature=request.temperature or self.config.temperature,
                max_tokens=request.max_tokens or self.config.max_tokens,
                top_p=request.top_p or self.config.top_p,
                top_k=request.top_k or self.config.top_k,
                system_prompt=request.system_prompt,
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
                    # Handle thinking deltas
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            if thinking_text:
                                yield StreamChunk(text=thinking_text, model=model, metadata={"type": "thinking"})
                            continue
                        elif delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield StreamChunk(text=text, model=model)
                            continue
                    # Fallback for non-thinking responses
                    text = self._extract_text_from_chunk(chunk_data, model)
                    if text:
                        yield StreamChunk(text=text, model=model)
                        
        except Exception:
            raise

    @Logger(exclude_args=[])
    async def send_message(self, request: MessageRequest) -> TextResponse:
        """Send a message in a conversation"""
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
        
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
        
        if request.use_cache and cache_key:
            cached, cache_source = self.cache.get(cache_key)
            if cached:
                Logger.note(f"Cache hit [{cache_key[:8]}] - {model}")
                result = self._deserialize_response(cached["data"], request.response_format, cached.get("metadata", {}))
                result.cache_source = cache_source
                result.cache_key = cache_key
                return result
        
        last_msg = request.messages[-1].content[:60] if request.messages else ""
        Logger.note(f"{model} | {len(request.messages)} messages | {last_msg}")
        
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
                started_at = datetime.now(timezone.utc).isoformat()
                t0 = time.monotonic()
                response = await self._invoke_model_with_retry(
                    modelId=model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                elapsed = round(time.monotonic() - t0, 3)
            
            response_body = json.loads(await response["body"].read())
            result = self._parse_response(response_body, model, request.response_format)
            result.timestamp = started_at
            result.elapsed_seconds = elapsed
            result.metadata["messages"] = [{"role": m.role, "content": m.content} for m in request.messages]
            result.metadata["response_format"] = request.response_format.model_json_schema() if request.response_format else None
            Logger.note(f"{result.input_tokens} in / {result.output_tokens} out | {result.text[:50]}")
            
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
            result.cache_key = cache_key
            return result
            
        except Exception:
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
                        
        except Exception:
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
            stop_reason = response_body.get("stop_reason", "")
            if stop_reason == "max_tokens" and response_format:
                raise ValueError("Bedrock truncated structured output (stop_reason=max_tokens)")
            # Check for tool use (structured output)
            content = response_body.get("content", [])
            if content and content[0].get("type") == "tool_use" and response_format:
                tool_input = content[0].get("input", {})
                structured_data = response_format.model_validate(tool_input)
                text = json.dumps(tool_input, indent=2)
            else:
                # Regular text response
                text = response_body["content"][0]["text"]
                structured_data = None
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

    def _parse_thinking_response(self, response_body: Dict[str, Any], model: str) -> TextResponse:
        """Parse a response that contains thinking blocks (extended thinking enabled)."""
        content = response_body.get("content", [])
        thinking_text = ""
        answer_text = ""

        for block in content:
            block_type = block.get("type", "")
            if block_type == "thinking":
                thinking_text += block.get("thinking", "")
            elif block_type == "text":
                answer_text += block.get("text", "")

        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Reasoning tokens: total output minus the text output approximation
        # The API doesn't separate them explicitly, so we report total output tokens
        # and store thinking text in metadata for transparency
        reasoning_tokens = output_tokens  # All output includes thinking in the count

        return TextResponse(
            text=answer_text,
            model=model,
            stop_reason=response_body.get("stop_reason", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            metadata={"thinking": thinking_text},
        )

    def _extract_text_from_chunk(self, chunk_data: Dict[str, Any], model: str) -> str:
        """Extract text from streaming chunk based on model type"""
        if "claude" in model.lower():
            if chunk_data.get("type") == "content_block_delta":
                return chunk_data.get("delta", {}).get("text", "")
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
            "reasoning_tokens": response.reasoning_tokens,
            "cached_tokens": response.cached_tokens,
            "timestamp": response.timestamp,
            "elapsed_seconds": response.elapsed_seconds,
            "structured_data": response.structured_data.model_dump() if response.structured_data else None,
        }
    
    def _deserialize_response(self, data: Dict[str, Any], response_format: Optional[Type[BaseModel]] = None, metadata: Optional[Dict[str, Any]] = None) -> TextResponse:
        """Deserialize cached data back to TextResponse"""
        structured_data = None
        if data.get("structured_data") and response_format:
            structured_data = response_format.model_validate(data["structured_data"])
        
        return TextResponse(
            text=data["text"],
            model=data["model"],
            stop_reason=data["stop_reason"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            reasoning_tokens=data.get("reasoning_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            timestamp=data.get("timestamp"),
            elapsed_seconds=data.get("elapsed_seconds"),
            metadata=metadata or {},
            structured_data=structured_data,
        )
