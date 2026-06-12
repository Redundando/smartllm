"""Main Bedrock LLM client wrapper"""

import json
import asyncio
import time
import inspect
import logging
from datetime import datetime, timezone
from typing import Optional, AsyncIterator, List, Dict, Any, Type, Callable
from pydantic import BaseModel
from logorator import Logger
from .config import BedrockConfig
from .capabilities import (
    ModelCapabilities,
    get_model_capabilities as _get_model_capabilities,
    supports_thinking as _supports_thinking,
    ADAPTIVE_EFFORT_LEVELS,
)
from ..models import (
    TextRequest,
    MessageRequest,
    TextResponse,
    StreamChunk,
)
from ..utils import pydantic_to_tool_schema, TwoLevelCache, retry_on_error
from ..utils.retry_utils import is_retryable_error, calculate_backoff
from ..defaults import BEDROCK_THINKING_BUDGET

logger = logging.getLogger('smartllm')

# --- Constants ---

# Extra tokens added to max_tokens when thinking budget exceeds the configured max.
# Ensures the model has room for both thinking and output text.
THINKING_BUDGET_HEADROOM = 4096

# Minimum thinking budget enforced by the Bedrock API.
MIN_THINKING_BUDGET = 1024

# Progress events fire every N estimated tokens (chars // CHARS_PER_TOKEN_ESTIMATE).
PROGRESS_TOKEN_THRESHOLD = 500

# Progress events also fire if this many seconds elapse without an event.
PROGRESS_TIME_THRESHOLD_SECONDS = 10

# Rough character-to-token ratio for estimating token counts from text length.
CHARS_PER_TOKEN_ESTIMATE = 4

# Maximum characters of the prompt included in progress events.
PROMPT_PREVIEW_LENGTH = 200

# Default Bedrock model quotas for concurrency limiting.
#
# Patterns are substring-matched against the model ID (lowercased), so they
# work for both bare foundation IDs ("anthropic.claude-...") and inference
# profile IDs with region prefixes ("eu.anthropic.claude-...", "us.…",
# "global.…"). Order matters: the first pattern that appears in the model ID
# wins, so list more specific patterns before more general ones.
DEFAULT_MODEL_QUOTAS = {
    # Anthropic Claude 4.x family (Sonnet / Opus) — modern inference profiles
    'claude-opus-4': {'rpm': 50, 'tpm': 200000, 'concurrent': 1},
    'claude-sonnet-4': {'rpm': 200, 'tpm': 400000, 'concurrent': 2},
    # Anthropic Claude 3.x family — legacy
    'claude-3-7-sonnet': {'rpm': 200, 'tpm': 400000, 'concurrent': 2},
    'claude-3-5-sonnet-v2': {'rpm': 10, 'tpm': 200000, 'concurrent': 1},
    'claude-3-5-sonnet': {'rpm': 200, 'tpm': 400000, 'concurrent': 2},
    'claude-3-5-haiku': {'rpm': 400, 'tpm': 400000, 'concurrent': 5},
    'claude-3-sonnet': {'rpm': 200, 'tpm': 400000, 'concurrent': 2},
    'claude-3-haiku': {'rpm': 400, 'tpm': 400000, 'concurrent': 5},
    'claude-3-opus': {'rpm': 50, 'tpm': 200000, 'concurrent': 1},
    # Other providers
    'llama': {'rpm': 500, 'tpm': 500000, 'concurrent': 5},
    'mistral': {'rpm': 300, 'tpm': 300000, 'concurrent': 3},
    'titan': {'rpm': 400, 'tpm': 400000, 'concurrent': 5},
    'nova': {'rpm': 400, 'tpm': 400000, 'concurrent': 5},
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
            boto_config = Config(
                max_pool_connections=pool_size,
                read_timeout=self.config.read_timeout,
                connect_timeout=self.config.connect_timeout,
            )

            session = aioboto3.Session()
            self.client = await session.client(
                "bedrock-runtime", config=boto_config, **creds
            ).__aenter__()
            self.models_client = await session.client(
                "bedrock", config=boto_config, **creds
            ).__aenter__()
            logger.info(
                f"Bedrock client initialized in region '{self.config.aws_region}' "
                f"(default model: {self.config.default_model})"
            )
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

    async def _invoke_model_with_retry(self, on_progress=None, retry_context=None, **kwargs):
        """Invoke model with retry logic and context-aware logging
        
        Args:
            on_progress: Optional callback for progress events (from request)
            retry_context: Dict with model, max_tokens for logging context
            **kwargs: Arguments passed to client.invoke_model
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
            return await self.client.invoke_model(**kwargs)
        
        try:
            return await _invoke()
        except Exception as e:
            self._maybe_log_region_hint(e, model)
            raise

    def _maybe_log_region_hint(self, error: Exception, model: str) -> None:
        """If the error looks like a region/model-availability mismatch, log a hint.

        Bedrock returns "The provided model identifier is invalid." for both
        genuinely invalid IDs and IDs that are valid but not deployed in the
        current region. The latter is the more common case for cross-region
        inference profile users (e.g. us.anthropic.* on a client bound to
        eu-north-1). We add a one-line hint so consumers don't waste time
        chasing the model ID when the region is the actual culprit.
        """
        msg = str(error)
        if "model identifier" not in msg.lower():
            return
        logger.warning(
            f"Bedrock rejected model '{model}' in region '{self.config.aws_region}'. "
            f"If the ID is correct, the region likely doesn't host this inference "
            f"profile. Set AWS_REGION/AWS_DEFAULT_REGION or pass aws_region= to match "
            f"a region where '{model}' is published."
        )

    def _make_retry_handler(self, on_progress, model, max_tokens):
        """Create retry handler that logs context and fires on_progress callback"""
        import inspect

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

    async def _consume_stream(
        self,
        response_stream,
        model: str,
        on_progress: Optional[Callable],
        start_time: float,
    ) -> dict:
        """Consume all chunks from a Bedrock response stream.

        Args:
            response_stream: The async iterable from invoke_model_with_response_stream.
            model: Model ID for context in progress events.
            on_progress: Optional callback for progress events.
            start_time: Monotonic clock start for elapsed_seconds calculation.

        Returns:
            Dict with keys: text, thinking_text, input_tokens, output_tokens,
            reasoning_tokens, stop_reason.
        """
        text_parts: list = []
        thinking_parts: list = []
        input_tokens: int = 0
        output_tokens: int = 0
        reasoning_tokens: int = 0
        stop_reason: str = ""

        # Progress tracking state
        last_text_progress_tokens: int = 0
        last_text_progress_time: float = start_time
        last_thinking_progress_tokens: int = 0
        last_thinking_progress_time: float = start_time

        async def _fire_progress(event: dict) -> None:
            """Fire a progress event, handling both async and sync callbacks."""
            if on_progress is None:
                return
            if inspect.iscoroutinefunction(on_progress):
                await on_progress(event)
            else:
                on_progress(event)

        async for event in response_stream:
            if "chunk" not in event:
                continue

            chunk_data = json.loads(event["chunk"]["bytes"])
            event_type = chunk_data.get("type", "")

            if event_type == "message_start":
                # Extract input_tokens from the message start metadata
                message = chunk_data.get("message", {})
                usage = message.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)

            elif event_type == "content_block_delta":
                delta = chunk_data.get("delta", {})
                delta_type = delta.get("type", "")

                if delta_type == "thinking_delta":
                    thinking_text = delta.get("thinking", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)

                        # Check thinking progress threshold
                        accumulated_thinking = "".join(thinking_parts)
                        current_thinking_tokens = len(accumulated_thinking) // CHARS_PER_TOKEN_ESTIMATE
                        now = time.monotonic()
                        tokens_since_last = current_thinking_tokens - last_thinking_progress_tokens
                        time_since_last = now - last_thinking_progress_time

                        if tokens_since_last >= PROGRESS_TOKEN_THRESHOLD or time_since_last >= PROGRESS_TIME_THRESHOLD_SECONDS:
                            await _fire_progress({
                                "event": "stream_thinking",
                                "thinking_tokens_so_far": current_thinking_tokens,
                                "thinking_text_so_far": accumulated_thinking,
                                "elapsed_seconds": round(now - start_time, 3),
                            })
                            last_thinking_progress_tokens = current_thinking_tokens
                            last_thinking_progress_time = now

                elif delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        text_parts.append(text)

                        # Check text progress threshold
                        accumulated_text = "".join(text_parts)
                        current_text_tokens = len(accumulated_text) // CHARS_PER_TOKEN_ESTIMATE
                        now = time.monotonic()
                        tokens_since_last = current_text_tokens - last_text_progress_tokens
                        time_since_last = now - last_text_progress_time

                        if tokens_since_last >= PROGRESS_TOKEN_THRESHOLD or time_since_last >= PROGRESS_TIME_THRESHOLD_SECONDS:
                            await _fire_progress({
                                "event": "stream_progress",
                                "text_tokens_so_far": current_text_tokens,
                                "text_so_far": accumulated_text,
                                "elapsed_seconds": round(now - start_time, 3),
                            })
                            last_text_progress_tokens = current_text_tokens
                            last_text_progress_time = now

            elif event_type == "message_delta":
                # Extract output token counts and stop reason from message_delta
                usage = chunk_data.get("usage", {})
                output_tokens = usage.get("output_tokens", output_tokens)
                reasoning_tokens = usage.get("reasoning_tokens", reasoning_tokens)
                delta = chunk_data.get("delta", {})
                stop_reason = delta.get("stop_reason", stop_reason)

            elif event_type == "message_stop":
                # Stream complete — stop_reason may also come here
                metrics = chunk_data.get("amazon-bedrock-invocationMetrics", {})
                if metrics:
                    # Prefer metrics values if available
                    input_tokens = metrics.get("inputTokenCount", input_tokens)
                    output_tokens = metrics.get("outputTokenCount", output_tokens)

        return {
            "text": "".join(text_parts),
            "thinking_text": "".join(thinking_parts),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "stop_reason": stop_reason,
        }

    async def generate_text_streamed(self, request: TextRequest) -> TextResponse:
        """Generate text using internal streaming, returning assembled response.

        Opens a streaming connection to Bedrock and consumes all chunks internally,
        assembling them into a single TextResponse. Fires progress callbacks during
        consumption. Respects concurrency semaphores, caching, and retry logic.

        Args:
            request: TextRequest with prompt and parameters.
                     Must NOT have response_format set.

        Returns:
            TextResponse with complete generated text, token counts, and metadata.

        Raises:
            ValueError: If request.response_format is not None.
        """
        # 1. Reject structured output — incompatible with streaming assembly
        if request.response_format is not None:
            raise ValueError(
                "generate_text_streamed does not support structured output (response_format). "
                "Use generate_text with the two-pass thinking approach as an alternative "
                "for structured output with large prompts."
            )

        if not self.client:
            await self._init_client()

        # 2. Resolve parameters using same logic as generate_text
        model = request.model or self.config.default_model
        thinking_requested = self._resolve_thinking_request(request, model)
        temperature = request.temperature if request.temperature is not None else 0
        on_progress = request.on_progress

        # 3. Cache key — same logic as generate_text
        cache_key = None
        if temperature == 0 or thinking_requested:
            cache_key = self._generate_cache_key(
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                system_prompt=request.system_prompt,
                response_format=None,  # Always None (structured output rejected above)
                reasoning_effort=request.reasoning_effort,
                budget_tokens=request.budget_tokens,
            )

        # 4. Handle clear_cache
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)

        # 5. Handle use_cache — return immediately on cache hit
        if request.use_cache and cache_key:
            cached, cache_source = self.cache.get(cache_key)
            if cached:
                # Fire cache_hit progress event
                if on_progress is not None:
                    event = {
                        "event": "cache_hit",
                        "cache_source": cache_source,
                        "model": model,
                    }
                    if inspect.iscoroutinefunction(on_progress):
                        await on_progress(event)
                    else:
                        on_progress(event)
                result = self._deserialize_response(cached["data"], None, cached.get("metadata", {}))
                result.cache_source = cache_source
                result.cache_key = cache_key
                return result

        # 6. Fire llm_started progress event
        if on_progress is not None:
            event = {
                "event": "llm_started",
                "model": model,
                "prompt": request.prompt[:PROMPT_PREVIEW_LENGTH],
                "provider": "bedrock",
            }
            if inspect.iscoroutinefunction(on_progress):
                await on_progress(event)
            else:
                on_progress(event)

        # 7. Build request body via the family-dispatching helper.
        # For Claude, this delegates to the capability-aware _build_claude_body
        # (handles thinking shape + sampling-param filtering).
        # For Llama/Mistral, the historical inline body is used.
        max_tokens = request.max_tokens or self.config.max_tokens
        body = self._build_request_body(
            model=model,
            prompt=request.prompt,
            temperature=temperature if not thinking_requested else None,
            max_tokens=max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=request.budget_tokens,
        )
        max_tokens = body.get("max_tokens", max_tokens)

        # 8. Acquire semaphore, open stream, consume, assemble response
        semaphore = self._get_semaphore(model)
        await semaphore.acquire()
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()

            # Retry loop: wraps stream open + consume
            last_error = None
            stream_result = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await self.client.invoke_model_with_response_stream(
                        body=json.dumps(body),
                        modelId=model,
                        accept="application/json",
                        contentType="application/json",
                    )

                    # Consume the stream
                    stream_result = await self._consume_stream(
                        response_stream=response["body"],
                        model=model,
                        on_progress=on_progress,
                        start_time=t0,
                    )
                    # Success — exit retry loop
                    break
                except Exception as e:
                    last_error = e
                    # Non-retryable or last attempt → raise
                    if not is_retryable_error(e) or attempt == self.config.max_retries:
                        self._maybe_log_region_hint(e, model)
                        raise
                    # Retryable: discard accumulated chunks, fire retry event, backoff
                    stream_result = None
                    delay = calculate_backoff(attempt, self.config.retry_delay, self.config.max_retry_delay)
                    # Fire retry progress event
                    if on_progress is not None:
                        retry_event = {
                            "event": "retry",
                            "attempt": attempt + 1,
                            "max_retries": self.config.max_retries,
                            "error": type(e).__name__,
                            "delay": round(delay, 1),
                        }
                        if inspect.iscoroutinefunction(on_progress):
                            await on_progress(retry_event)
                        else:
                            on_progress(retry_event)
                    logger.warning(
                        f"Stream retry {attempt + 1}/{self.config.max_retries} after "
                        f"{type(e).__name__} on model {model}. Waiting {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    # Reset start time for next attempt
                    t0 = time.monotonic()

            elapsed = round(time.monotonic() - t0, 3)

            # 9. Assemble TextResponse
            metadata = {}
            if stream_result["thinking_text"]:
                metadata["thinking"] = stream_result["thinking_text"]

            result = TextResponse(
                text=stream_result["text"],
                model=model,
                stop_reason=stream_result["stop_reason"],
                input_tokens=stream_result["input_tokens"],
                output_tokens=stream_result["output_tokens"],
                reasoning_tokens=stream_result["reasoning_tokens"],
                cached_tokens=0,
                timestamp=started_at,
                elapsed_seconds=elapsed,
                metadata=metadata,
                cache_source="miss",
            )

            # 10. Write to cache if cache-eligible
            if cache_key:
                cache_metadata = {
                    "prompt": request.prompt,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt": request.system_prompt,
                    "response_format": None,
                    "top_p": request.top_p or self.config.top_p,
                    "top_k": request.top_k or self.config.top_k,
                    "reasoning_effort": request.reasoning_effort,
                    "budget_tokens": request.budget_tokens,
                }
                self.cache.set(cache_key, self._serialize_response(result), cache_metadata)
            result.cache_key = cache_key

            # 11. Fire llm_done progress event
            if on_progress is not None:
                event = {
                    "event": "llm_done",
                    "input_tokens": stream_result["input_tokens"],
                    "output_tokens": stream_result["output_tokens"],
                    "reasoning_tokens": stream_result["reasoning_tokens"],
                    "elapsed_seconds": elapsed,
                }
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(event)
                else:
                    on_progress(event)

            return result

        except Exception as e:
            # Fire error progress event
            text_so_far = ""
            tokens_so_far = 0
            # Try to get accumulated text from locals if available
            if 'stream_result' in locals() and stream_result is not None:
                text_so_far = stream_result.get("text", "")
                tokens_so_far = len(text_so_far) // CHARS_PER_TOKEN_ESTIMATE
            
            if on_progress is not None:
                error_event = {
                    "event": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "text_so_far": text_so_far,
                    "tokens_so_far": tokens_so_far,
                }
                try:
                    if inspect.iscoroutinefunction(on_progress):
                        await on_progress(error_event)
                    else:
                        on_progress(error_event)
                except Exception:
                    pass  # Don't let callback errors mask the original error

            # Attach context to exception
            e.text_so_far = text_so_far
            e.tokens_so_far = tokens_so_far
            raise
        finally:
            semaphore.release()

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

    @staticmethod
    def get_model_capabilities(model_id: str) -> ModelCapabilities:
        """Inspect what a Claude model on Bedrock will accept in the request body.

        Useful for callers who route across many models and want to adapt their
        request shape (e.g. drop `temperature`, downgrade thinking to adaptive)
        before calling. For non-Claude or unrecognised IDs, returns a permissive
        default.
        """
        return _get_model_capabilities(model_id)

    @staticmethod
    def supports_thinking(model_id: str) -> bool:
        """Whether `model_id` supports any form of extended thinking."""
        return _supports_thinking(model_id)

    @staticmethod
    def _validate_tool_input(tool_input: Any, response_format: Type[BaseModel]) -> BaseModel:
        """Validate tool-use input against a Pydantic model, tolerantly.

        Bedrock occasionally returns complex fields (lists, nested objects) as
        JSON-encoded strings instead of native JSON values, particularly on
        non-English content with the two-pass thinking + structure flow. This
        method first attempts a strict validation; on failure it walks the dict
        once and tries `json.loads` on any string values that look like a JSON
        array or object, then retries.
        """
        # Top-level may itself be a string — handle that first.
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except (ValueError, json.JSONDecodeError):
                pass  # let the validator raise the right error

        try:
            return response_format.model_validate(tool_input)
        except Exception:
            # Retry once after coercing any stringified JSON values.
            if not isinstance(tool_input, dict):
                raise
            coerced: Dict[str, Any] = {}
            for k, v in tool_input.items():
                if isinstance(v, str):
                    stripped = v.lstrip()
                    if stripped.startswith("[") or stripped.startswith("{"):
                        try:
                            coerced[k] = json.loads(v)
                            continue
                        except (ValueError, json.JSONDecodeError):
                            pass
                coerced[k] = v
            return response_format.model_validate(coerced)

    def _resolve_thinking_request(self, request: TextRequest, model: str) -> bool:
        """Whether the caller is asking for extended thinking AND the model supports it.

        The actual thinking shape (manual budget vs. adaptive effort) is decided
        inside `_build_claude_body`; this method only answers the dispatch
        question "do we route to a thinking-capable code path?".
        """
        wants = request.reasoning_effort is not None or request.budget_tokens is not None
        if not wants:
            return False
        caps = _get_model_capabilities(model)
        if caps.thinking_mode == "none":
            logger.warning(
                f"Model {model} does not support extended thinking; "
                f"reasoning_effort/budget_tokens will be ignored."
            )
            return False
        return True

    def _build_claude_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        force_tool_use: bool = True,
    ) -> Dict[str, Any]:
        """Build a Bedrock-Anthropic request body, respecting per-model capabilities.

        Sampling parameters and thinking configuration are included only if the
        target model accepts them. Callers pass everything they have; the helper
        decides what makes it into the body.

        For thinking-capable models, the helper accepts both `reasoning_effort`
        (preferred) and `budget_tokens`. The mode (manual budget vs. adaptive
        effort) is determined by the model's capabilities.

        Args:
            model: The Bedrock model ID or inference profile ID.
            messages: Anthropic-shaped messages list.
            max_tokens: Cap on output tokens. Auto-extended for manual thinking
                budgets that meet or exceed this value.
            temperature/top_p/top_k: Sampling controls; dropped silently for
                models that reject them (with a warning).
            system_prompt: Optional system prompt.
            response_format: Optional Pydantic model for structured output via
                forced tool use.
            reasoning_effort: "low" | "medium" | "high". Used directly on
                adaptive models; mapped via `BEDROCK_THINKING_BUDGET` on
                manual-budget models.
            budget_tokens: Explicit thinking budget. Used on manual-budget
                models; logged-and-mapped to nearest effort on adaptive ones.
            force_tool_use: When True, sets `tool_choice` to force the response
                format tool. False is reserved for cases where tools should be
                offered without forcing.

        Returns:
            A dict ready to be JSON-serialized as the Bedrock request body.
        """
        caps = _get_model_capabilities(model)

        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # --- Sampling controls ---
        if temperature is not None:
            if caps.accepts_temperature:
                body["temperature"] = temperature
            else:
                logger.warning(f"Model {model} does not accept 'temperature'; ignored.")
        if top_p is not None:
            if caps.accepts_top_p_top_k:
                body["top_p"] = top_p
            else:
                logger.warning(f"Model {model} does not accept 'top_p'; ignored.")
        if top_k is not None:
            if caps.accepts_top_p_top_k:
                body["top_k"] = top_k
            else:
                logger.warning(f"Model {model} does not accept 'top_k'; ignored.")

        # --- System prompt ---
        if system_prompt:
            body["system"] = system_prompt

        # --- Extended thinking ---
        wants_thinking = reasoning_effort is not None or budget_tokens is not None
        if wants_thinking and caps.thinking_mode != "none":
            if caps.thinking_mode == "manual_budget":
                resolved_budget = self._resolve_manual_budget(reasoning_effort, budget_tokens)
                body["thinking"] = {"type": "enabled", "budget_tokens": resolved_budget}
                if resolved_budget >= body["max_tokens"]:
                    body["max_tokens"] = resolved_budget + THINKING_BUDGET_HEADROOM
            else:  # adaptive_effort
                effort = self._resolve_adaptive_effort(reasoning_effort, budget_tokens, model)
                body["thinking"] = {"type": "adaptive"}
                body["output_config"] = {"effort": effort}
        elif wants_thinking:
            # caps.thinking_mode == "none"; already warned in _resolve_thinking_request
            # but defensive log here too in case caller bypassed the dispatch helper
            logger.warning(f"Model {model} does not support thinking; thinking parameters dropped.")

        # --- Structured output via forced tool use ---
        if response_format is not None:
            tool_schema = pydantic_to_tool_schema(response_format)
            body["tools"] = [tool_schema]
            if force_tool_use:
                body["tool_choice"] = {"type": "tool", "name": tool_schema["name"]}

        return body

    @staticmethod
    def _resolve_manual_budget(reasoning_effort: Optional[str], budget_tokens: Optional[int]) -> int:
        """Resolve thinking budget for models using `thinking.type=enabled`."""
        if budget_tokens is not None:
            return max(budget_tokens, MIN_THINKING_BUDGET)
        # reasoning_effort is the only remaining input
        budget = BEDROCK_THINKING_BUDGET.get(reasoning_effort)
        if budget is None:
            raise ValueError(
                f"Invalid reasoning_effort '{reasoning_effort}'. "
                f"Must be one of: {', '.join(BEDROCK_THINKING_BUDGET.keys())}"
            )
        return max(budget, MIN_THINKING_BUDGET)

    @staticmethod
    def _resolve_adaptive_effort(reasoning_effort: Optional[str], budget_tokens: Optional[int], model: str) -> str:
        """Resolve effort label for models using `thinking.type=adaptive`."""
        if reasoning_effort is not None:
            if reasoning_effort not in ADAPTIVE_EFFORT_LEVELS:
                raise ValueError(
                    f"Invalid reasoning_effort '{reasoning_effort}' for adaptive thinking on {model}. "
                    f"Must be one of: {', '.join(ADAPTIVE_EFFORT_LEVELS)}"
                )
            return reasoning_effort
        # Only budget_tokens given; map to nearest effort label.
        # Mapping uses the same thresholds as the manual-budget reasoning_effort table.
        effort = "low"
        if budget_tokens >= BEDROCK_THINKING_BUDGET["high"]:
            effort = "high"
        elif budget_tokens >= BEDROCK_THINKING_BUDGET["medium"]:
            effort = "medium"
        logger.warning(
            f"Model {model} uses adaptive thinking; mapped budget_tokens={budget_tokens} -> effort={effort}."
        )
        return effort

    @Logger(exclude_args=[])
    async def generate_text(self, request: TextRequest) -> TextResponse:
        """Generate text from a prompt"""
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        thinking_requested = self._resolve_thinking_request(request, model)
        # If no temperature specified, use 0 (deterministic + cacheable)
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key for this specific request
        cache_key = None
        if (temperature == 0 or thinking_requested) and not request.stream:
            cache_key = self._generate_cache_key(
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None,
                reasoning_effort=request.reasoning_effort,
                budget_tokens=request.budget_tokens,
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
        Logger.note(f"{model} | temp={temperature} | thinking={'on' if thinking_requested else 'off'} | {prompt_preview}")
        
        # Two-pass approach when both thinking and structured output are requested
        if thinking_requested and request.response_format:
            result = await self._generate_with_thinking_and_structure(
                request=request,
                model=model,
                temperature=temperature,
            )
        elif thinking_requested:
            result = await self._generate_with_thinking(
                request=request,
                model=model,
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
                "budget_tokens": request.budget_tokens,
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
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            response_format=request.response_format,
        )

        semaphore = self._get_semaphore(model)
        async with semaphore:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response = await self._invoke_model_with_retry(
                on_progress=request.on_progress,
                retry_context={"model": model, "max_tokens": request.max_tokens or self.config.max_tokens},
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

    async def _generate_with_thinking(self, request: TextRequest, model: str) -> TextResponse:
        """Generation with extended thinking, no structured output."""
        max_tokens = request.max_tokens or self.config.max_tokens
        body = self._build_claude_body(
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=max_tokens,
            system_prompt=request.system_prompt,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=request.budget_tokens,
        )
        max_tokens = body["max_tokens"]  # may have been bumped to fit thinking budget

        semaphore = self._get_semaphore(model)
        async with semaphore:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response = await self._invoke_model_with_retry(
                on_progress=request.on_progress,
                retry_context={"model": model, "max_tokens": max_tokens},
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
        self, request: TextRequest, model: str, temperature: float
    ) -> TextResponse:
        """Two-pass: thinking first, then structured extraction.
        
        Pass 1: Extended thinking to reason through the prompt.
        Pass 2: Forced tool use to extract structured output from pass 1 result.
        """
        # --- Pass 1: Think ---
        max_tokens = request.max_tokens or self.config.max_tokens
        body_pass1 = self._build_claude_body(
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=max_tokens,
            system_prompt=request.system_prompt,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=request.budget_tokens,
        )
        max_tokens = body_pass1["max_tokens"]

        semaphore = self._get_semaphore(model)
        async with semaphore:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response1 = await self._invoke_model_with_retry(
                on_progress=request.on_progress,
                retry_context={"model": model, "max_tokens": max_tokens},
                modelId=model,
                body=json.dumps(body_pass1),
                contentType="application/json",
            )

            response_body1 = json.loads(await response1["body"].read())
            pass1_result = self._parse_thinking_response(response_body1, model)
            Logger.note(f"Pass 1 (thinking): {pass1_result.input_tokens} in / {pass1_result.output_tokens} out (thinking={pass1_result.reasoning_tokens})")

            # --- Pass 2: Structure extraction ---
            # No thinking on pass 2; force the response_format tool. The hint about
            # native arrays/objects addresses an observed Bedrock quirk where
            # complex fields occasionally arrive as JSON-encoded strings.
            extraction_prompt = (
                "Extract the content from the following text into the required structured format. "
                "Map the information to the schema fields as accurately as possible. "
                "Do not add, invent, or omit any information — use only what is provided. "
                "Return list and object values as native JSON arrays/objects, never as JSON-encoded strings.\n\n"
                f"---\n{pass1_result.text}\n---"
            )
            body_pass2 = self._build_claude_body(
                model=model,
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=0,
                response_format=request.response_format,
            )

            response2 = await self._invoke_model_with_retry(
                on_progress=request.on_progress,
                retry_context={"model": model, "max_tokens": request.max_tokens or self.config.max_tokens},
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
        thinking_requested = self._resolve_thinking_request(request, model)

        # Dispatch by model family — supports Claude (with thinking) plus
        # Llama/Mistral via the historical body shapes.
        body = self._build_request_body(
            model=model,
            prompt=request.prompt,
            temperature=None if thinking_requested else (request.temperature if request.temperature is not None else self.config.temperature),
            max_tokens=request.max_tokens or self.config.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=request.budget_tokens,
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
                        
        except Exception as e:
            self._maybe_log_region_hint(e, model)
            raise

    @Logger(exclude_args=[])
    async def send_message(self, request: MessageRequest) -> TextResponse:
        """Send a message in a conversation"""
        if not self.client:
            await self._init_client()
            
        model = request.model or self.config.default_model
        thinking_requested = self._resolve_thinking_request(request, model)
        # If no temperature specified, use 0 (deterministic + cacheable)
        temperature = request.temperature if request.temperature is not None else 0
        
        # Generate cache key for this specific request
        cache_key = None
        if (temperature == 0 or thinking_requested) and not request.stream:
            messages_str = json.dumps([{"role": m.role, "content": m.content} for m in request.messages])
            cache_key = self._generate_cache_key(
                model=model,
                messages=messages_str,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None,
                reasoning_effort=request.reasoning_effort,
                budget_tokens=request.budget_tokens,
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
        Logger.note(f"{model} | {len(request.messages)} messages | thinking={'on' if thinking_requested else 'off'} | {last_msg}")
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        body = self._build_claude_body(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens or self.config.max_tokens,
            temperature=None if thinking_requested else temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            response_format=request.response_format if "claude" in model.lower() else None,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=request.budget_tokens,
        )

        try:
            semaphore = self._get_semaphore(model)
            async with semaphore:
                started_at = datetime.now(timezone.utc).isoformat()
                t0 = time.monotonic()
                response = await self._invoke_model_with_retry(
                    on_progress=request.on_progress,
                    retry_context={"model": model, "max_tokens": request.max_tokens or self.config.max_tokens},
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
        thinking_requested = self._resolve_thinking_request(request, model)

        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        body = self._build_claude_body(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens or self.config.max_tokens,
            temperature=None if thinking_requested else (request.temperature if request.temperature is not None else self.config.temperature),
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=request.budget_tokens,
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
            self._maybe_log_region_hint(e, model)
            raise

    def _build_request_body(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float],
        max_tokens: int,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        budget_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a request body for text generation, dispatching by model family.

        For Claude models we delegate to the capability-aware `_build_claude_body`
        so that newer models (Opus 4.7+, etc.) get the right shape. For Llama and
        Mistral we keep the historical inline construction. Thinking parameters
        are forwarded to Claude and ignored elsewhere (Llama/Mistral on Bedrock
        don't expose extended thinking).

        `temperature=None` is interpreted as "omit if the model rejects it,
        otherwise fall back to the config default" — Llama/Mistral always
        require a temperature, so it's resolved against `self.config.temperature`
        for those branches.
        """
        if "claude" in model.lower():
            return self._build_claude_body(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                system_prompt=system_prompt,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                budget_tokens=budget_tokens,
            )
        if reasoning_effort is not None or budget_tokens is not None:
            logger.warning(
                f"Model {model} does not support extended thinking; "
                f"reasoning_effort/budget_tokens will be ignored."
            )
        # Llama/Mistral/generic require a numeric temperature.
        resolved_temperature = temperature if temperature is not None else self.config.temperature
        resolved_top_p = top_p if top_p is not None else self.config.top_p
        if "llama" in model.lower():
            return {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": resolved_temperature,
                "top_p": resolved_top_p,
            }
        if "mistral" in model.lower():
            return {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": resolved_temperature,
                "top_p": resolved_top_p,
            }
        # Default/generic format
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": resolved_temperature,
            "top_p": resolved_top_p,
        }

    def _parse_response(self, response_body: Dict[str, Any], model: str, response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Parse response based on model type"""
        if "claude" in model.lower():
            stop_reason = response_body.get("stop_reason", "")
            if stop_reason == "max_tokens" and response_format:
                raise ValueError("Bedrock truncated structured output (stop_reason=max_tokens)")
            # Check for tool use (structured output)
            content = response_body.get("content", [])
            tool_block = next((b for b in content if b.get("type") == "tool_use"), None)
            if tool_block is not None and response_format:
                tool_input = tool_block.get("input", {})
                structured_data = self._validate_tool_input(tool_input, response_format)
                text = json.dumps(tool_input, indent=2) if isinstance(tool_input, dict) else str(tool_input)
            else:
                # Regular text response
                text_block = next((b for b in content if b.get("type") == "text"), None)
                text = text_block.get("text", "") if text_block else ""
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
