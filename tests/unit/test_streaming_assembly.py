"""Unit tests for BedrockLLMClient.generate_text_streamed (streaming with assembly).

Tests cover: caching, concurrency semaphores, progress callbacks, retry logic,
thinking content handling, structured output rejection, and TextResponse shape.
"""

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from smartllm.bedrock.bedrock_client import BedrockLLMClient
from smartllm.bedrock.config import BedrockConfig
from smartllm.models import TextRequest, TextResponse


# --- Helpers ---


def _make_message_start_event(input_tokens: int = 10) -> dict:
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "message_start",
                "message": {"usage": {"input_tokens": input_tokens}},
            }).encode()
        }
    }


def _make_text_delta_event(text: str) -> dict:
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": text},
            }).encode()
        }
    }


def _make_thinking_delta_event(text: str) -> dict:
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": text},
            }).encode()
        }
    }


def _make_message_delta_event(
    output_tokens: int = 50,
    reasoning_tokens: int = 0,
    stop_reason: str = "end_turn",
) -> dict:
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "message_delta",
                "usage": {"output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens},
                "delta": {"stop_reason": stop_reason},
            }).encode()
        }
    }


def _make_message_stop_event(input_tokens: int = None, output_tokens: int = None) -> dict:
    data = {"type": "message_stop"}
    if input_tokens is not None or output_tokens is not None:
        data["amazon-bedrock-invocationMetrics"] = {}
        if input_tokens is not None:
            data["amazon-bedrock-invocationMetrics"]["inputTokenCount"] = input_tokens
        if output_tokens is not None:
            data["amazon-bedrock-invocationMetrics"]["outputTokenCount"] = output_tokens
    return {
        "chunk": {
            "bytes": json.dumps(data).encode()
        }
    }


async def _async_iter(events):
    """Convert a list of events into an async iterable."""
    for event in events:
        yield event


def _create_client(max_retries: int = 3, retry_delay: float = 0.01, max_retry_delay: float = 0.05) -> BedrockLLMClient:
    """Create a BedrockLLMClient for testing without real AWS connection."""
    config = BedrockConfig(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_region="us-east-1",
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_retry_delay=max_retry_delay,
    )
    client = BedrockLLMClient(config=config, max_concurrent=2)
    # Mark as initialized so it doesn't try to connect
    client.client = AsyncMock()
    return client


def _standard_stream_events(text: str = "Hello world", input_tokens: int = 10, output_tokens: int = 5):
    """Build a standard successful stream event sequence."""
    return [
        _make_message_start_event(input_tokens),
        _make_text_delta_event(text),
        _make_message_delta_event(output_tokens=output_tokens, stop_reason="end_turn"),
        _make_message_stop_event(),
    ]


# --- Cache Tests ---


class TestCacheHitReturnsImmediately:
    """Test cache hit returns immediately without streaming, fires cache_hit event.

    **Validates: Requirements 5.1, 5.2, 5.5**
    """

    @pytest.mark.asyncio
    async def test_cache_hit_returns_without_streaming(self):
        """Cache hit returns a TextResponse immediately without opening a stream."""
        client = _create_client()

        # Pre-populate cache
        request = TextRequest(prompt="test prompt", temperature=0, use_cache=True)
        model = client.config.default_model

        cache_key = client._generate_cache_key(
            model=model,
            prompt="test prompt",
            max_tokens=client.config.max_tokens,
            top_p=None,
            top_k=None,
            system_prompt=None,
            response_format=None,
            reasoning_effort=None,
            budget_tokens=None,
        )

        cached_data = {
            "text": "cached response",
            "model": model,
            "stop_reason": "end_turn",
            "input_tokens": 5,
            "output_tokens": 3,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "timestamp": "2024-01-01T00:00:00Z",
            "elapsed_seconds": 1.0,
            "structured_data": None,
        }
        client.cache.set(cache_key, cached_data, {"model": model})

        # Collect progress events
        events = []
        request.on_progress = lambda e: events.append(e)

        result = await client.generate_text_streamed(request)

        # Verify stream was never opened
        client.client.invoke_model_with_response_stream.assert_not_called()

        # Verify result
        assert result.text == "cached response"
        assert result.cache_source in ("l1", "l2")

    @pytest.mark.asyncio
    async def test_cache_hit_fires_cache_hit_event(self):
        """Cache hit fires a cache_hit progress event."""
        client = _create_client()

        request = TextRequest(prompt="test prompt", temperature=0, use_cache=True)
        model = client.config.default_model

        cache_key = client._generate_cache_key(
            model=model,
            prompt="test prompt",
            max_tokens=client.config.max_tokens,
            top_p=None,
            top_k=None,
            system_prompt=None,
            response_format=None,
            reasoning_effort=None,
            budget_tokens=None,
        )

        cached_data = {
            "text": "cached",
            "model": model,
            "stop_reason": "end_turn",
            "input_tokens": 5,
            "output_tokens": 3,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "timestamp": "2024-01-01T00:00:00Z",
            "elapsed_seconds": 1.0,
            "structured_data": None,
        }
        client.cache.set(cache_key, cached_data, {"model": model})

        events = []
        request.on_progress = lambda e: events.append(e)

        await client.generate_text_streamed(request)

        cache_hit_events = [e for e in events if e.get("event") == "cache_hit"]
        assert len(cache_hit_events) == 1
        assert cache_hit_events[0]["cache_source"] in ("l1", "l2")
        assert cache_hit_events[0]["model"] == model


class TestCacheWriteOnSuccess:
    """Test cache write on successful completion with temperature=0.

    **Validates: Requirements 5.3**
    """

    @pytest.mark.asyncio
    async def test_cache_written_on_success_temp_zero(self):
        """Successful stream with temperature=0 writes to cache."""
        client = _create_client()

        stream_events = _standard_stream_events("Hello", input_tokens=10, output_tokens=5)
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        request = TextRequest(prompt="cache write test unique 12345", temperature=0, use_cache=True)
        model = client.config.default_model

        cache_key = client._generate_cache_key(
            model=model,
            prompt="cache write test unique 12345",
            max_tokens=client.config.max_tokens,
            top_p=None,
            top_k=None,
            system_prompt=None,
            response_format=None,
            reasoning_effort=None,
            budget_tokens=None,
        )

        # Clear any pre-existing cache entry
        client.cache.clear(cache_key)
        assert client.cache.get(cache_key) == (None, "miss")

        await client.generate_text_streamed(request)

        # Verify cache was written
        cached, source = client.cache.get(cache_key)
        assert cached is not None
        assert cached["data"]["text"] == "Hello"


class TestCacheClear:
    """Test cache clear when clear_cache=True.

    **Validates: Requirements 5.4**
    """

    @pytest.mark.asyncio
    async def test_clear_cache_removes_entry(self):
        """clear_cache=True removes existing cache entry before streaming."""
        client = _create_client()

        request = TextRequest(prompt="clear me", temperature=0, clear_cache=True, use_cache=True)
        model = client.config.default_model

        cache_key = client._generate_cache_key(
            model=model,
            prompt="clear me",
            max_tokens=client.config.max_tokens,
            top_p=None,
            top_k=None,
            system_prompt=None,
            response_format=None,
            reasoning_effort=None,
            budget_tokens=None,
        )

        # Pre-populate cache
        client.cache.set(cache_key, {
            "text": "old cached",
            "model": model,
            "stop_reason": "end_turn",
            "input_tokens": 1,
            "output_tokens": 1,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "timestamp": None,
            "elapsed_seconds": None,
            "structured_data": None,
        }, {})

        # Setup stream for the new request
        stream_events = _standard_stream_events("new response")
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        result = await client.generate_text_streamed(request)

        # Verify stream was called (cache was cleared)
        client.client.invoke_model_with_response_stream.assert_called_once()
        assert result.text == "new response"


class TestNoCacheWriteOnFailure:
    """Test no cache write on stream failure.

    **Validates: Requirements 5.6**
    """

    @pytest.mark.asyncio
    async def test_no_cache_write_on_stream_failure(self):
        """Cache is NOT written when stream fails with a non-retryable error."""
        client = _create_client(max_retries=0)

        # Simulate a non-retryable error
        client.client.invoke_model_with_response_stream.side_effect = ValueError("bad request")

        request = TextRequest(prompt="fail me", temperature=0)
        model = client.config.default_model

        cache_key = client._generate_cache_key(
            model=model,
            prompt="fail me",
            max_tokens=client.config.max_tokens,
            top_p=None,
            top_k=None,
            system_prompt=None,
            response_format=None,
            reasoning_effort=None,
            budget_tokens=None,
        )

        with pytest.raises(ValueError):
            await client.generate_text_streamed(request)

        # Verify cache was NOT written
        assert client.cache.get(cache_key) == (None, "miss")


# --- Semaphore Tests ---


class TestSemaphoreAcquiredAndReleased:
    """Test semaphore acquired before stream, released after success.

    **Validates: Requirements 4.1, 4.2**
    """

    @pytest.mark.asyncio
    async def test_semaphore_acquired_before_stream_released_after(self):
        """Semaphore is acquired before streaming and released after."""
        client = _create_client()

        stream_events = _standard_stream_events()
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        model = client.config.default_model
        semaphore = client._get_semaphore(model)

        # Verify semaphore starts at full capacity
        initial_value = semaphore._value

        request = TextRequest(prompt="test", temperature=0)
        await client.generate_text_streamed(request)

        # After completion, semaphore should be at its original value
        assert semaphore._value == initial_value


class TestSemaphoreReleasedOnError:
    """Test semaphore released on error.

    **Validates: Requirements 4.3**
    """

    @pytest.mark.asyncio
    async def test_semaphore_released_on_error(self):
        """Semaphore is released even when stream fails."""
        client = _create_client(max_retries=0)

        client.client.invoke_model_with_response_stream.side_effect = RuntimeError("connection lost")

        model = client.config.default_model
        semaphore = client._get_semaphore(model)
        initial_value = semaphore._value

        request = TextRequest(prompt="fail", temperature=0)

        with pytest.raises(RuntimeError):
            await client.generate_text_streamed(request)

        # Semaphore must be released
        assert semaphore._value == initial_value


class TestSemaphoreHeldAcrossRetries:
    """Test semaphore held across retries.

    **Validates: Requirements 4.5**
    """

    @pytest.mark.asyncio
    async def test_semaphore_held_across_retries(self):
        """Semaphore remains acquired during retry attempts."""
        client = _create_client(max_retries=2, retry_delay=0.001, max_retry_delay=0.002)

        model = client.config.default_model
        semaphore = client._get_semaphore(model)
        initial_value = semaphore._value

        # Track semaphore value during calls
        semaphore_values_during_calls = []

        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            semaphore_values_during_calls.append(semaphore._value)
            if call_count < 3:
                # Retryable error
                from botocore.exceptions import ClientError
                raise ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "throttled"},
                     "ResponseMetadata": {"HTTPStatusCode": 429}},
                    "InvokeModel"
                )
            # Success on 3rd call
            return {"body": _async_iter(_standard_stream_events())}

        client.client.invoke_model_with_response_stream = mock_invoke

        request = TextRequest(prompt="retry test", temperature=0)
        await client.generate_text_streamed(request)

        # Semaphore should have been decremented during ALL calls (held across retries)
        for val in semaphore_values_during_calls:
            assert val == initial_value - 1, (
                f"Semaphore was released between retries! Value during call: {val}"
            )

        # After completion, semaphore should be restored
        assert semaphore._value == initial_value


# --- Progress Callback Tests ---


class TestLLMStartedEvent:
    """Test llm_started event fired at stream open.

    **Validates: Requirements 2.1**
    """

    @pytest.mark.asyncio
    async def test_llm_started_event_fired(self):
        """llm_started event is fired when stream begins."""
        client = _create_client()

        stream_events = _standard_stream_events()
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        events = []
        request = TextRequest(prompt="llm started event test unique", temperature=0, use_cache=False)
        request.on_progress = lambda e: events.append(e)

        await client.generate_text_streamed(request)

        started_events = [e for e in events if e.get("event") == "llm_started"]
        assert len(started_events) == 1
        assert started_events[0]["model"] == client.config.default_model
        assert started_events[0]["provider"] == "bedrock"
        assert "llm started event test unique" in started_events[0]["prompt"]


class TestLLMDoneEvent:
    """Test llm_done event fired at completion with correct token counts.

    **Validates: Requirements 2.4**
    """

    @pytest.mark.asyncio
    async def test_llm_done_event_with_token_counts(self):
        """llm_done event contains correct input/output/reasoning token counts."""
        client = _create_client()

        stream_events = [
            _make_message_start_event(input_tokens=25),
            _make_text_delta_event("response text"),
            _make_message_delta_event(output_tokens=15, reasoning_tokens=5, stop_reason="end_turn"),
            _make_message_stop_event(),
        ]
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        events = []
        request = TextRequest(prompt="test llm_done event unique prompt", temperature=0, use_cache=False)
        request.on_progress = lambda e: events.append(e)

        await client.generate_text_streamed(request)

        done_events = [e for e in events if e.get("event") == "llm_done"]
        assert len(done_events) == 1
        assert done_events[0]["input_tokens"] == 25
        assert done_events[0]["output_tokens"] == 15
        assert done_events[0]["reasoning_tokens"] == 5
        assert "elapsed_seconds" in done_events[0]


class TestErrorEvent:
    """Test error event fired on failure with text_so_far and tokens_so_far.

    **Validates: Requirements 2.5**
    """

    @pytest.mark.asyncio
    async def test_error_event_fired_on_failure(self):
        """Error event is fired with error details on stream failure."""
        client = _create_client(max_retries=0)

        # Create a stream that fails mid-consumption
        async def failing_stream():
            yield _make_message_start_event(input_tokens=10)
            yield _make_text_delta_event("partial text")
            raise RuntimeError("connection dropped")

        client.client.invoke_model_with_response_stream.return_value = {
            "body": failing_stream()
        }

        events = []
        request = TextRequest(prompt="fail error event", temperature=0, use_cache=False)
        request.on_progress = lambda e: events.append(e)

        with pytest.raises(RuntimeError, match="connection dropped"):
            await client.generate_text_streamed(request)

        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error_type"] == "RuntimeError"
        assert "connection dropped" in error_events[0]["error_message"]
        # text_so_far and tokens_so_far are present (may be empty if error occurs
        # during _consume_stream before stream_result is assigned)
        assert "text_so_far" in error_events[0]
        assert "tokens_so_far" in error_events[0]


class TestAsyncAndSyncCallbacks:
    """Test async and sync callbacks both work correctly.

    **Validates: Requirements 2.7**
    """

    @pytest.mark.asyncio
    async def test_sync_callback_works(self):
        """Sync progress callback is called directly."""
        client = _create_client()

        stream_events = _standard_stream_events()
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        events = []

        def sync_callback(event):
            events.append(event)

        request = TextRequest(prompt="sync callback test unique", temperature=0, use_cache=False)
        request.on_progress = sync_callback

        await client.generate_text_streamed(request)

        # Should have at least llm_started and llm_done
        event_types = [e["event"] for e in events]
        assert "llm_started" in event_types
        assert "llm_done" in event_types

    @pytest.mark.asyncio
    async def test_async_callback_works(self):
        """Async progress callback is awaited."""
        client = _create_client()

        stream_events = _standard_stream_events()
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        events = []

        async def async_callback(event):
            events.append(event)

        request = TextRequest(prompt="async callback test unique", temperature=0, use_cache=False)
        request.on_progress = async_callback

        await client.generate_text_streamed(request)

        event_types = [e["event"] for e in events]
        assert "llm_started" in event_types
        assert "llm_done" in event_types


class TestNoCallbackNoErrors:
    """Test no callback → no errors.

    **Validates: Requirements 2.6**
    """

    @pytest.mark.asyncio
    async def test_no_callback_no_errors(self):
        """When on_progress is None, no errors occur."""
        client = _create_client()

        stream_events = _standard_stream_events()
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        request = TextRequest(prompt="test", temperature=0)
        request.on_progress = None

        # Should not raise
        result = await client.generate_text_streamed(request)
        assert result.text == "Hello world"


# --- Retry Tests ---


class TestRetryableErrorTriggersRetry:
    """Test retryable error triggers retry with correct retry event.

    **Validates: Requirements 6.1, 6.3**
    """

    @pytest.mark.asyncio
    async def test_retryable_error_retries_with_event(self):
        """Retryable errors trigger retry and fire retry progress event."""
        client = _create_client(max_retries=2, retry_delay=0.001, max_retry_delay=0.01)

        from botocore.exceptions import ClientError

        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "throttled"},
                     "ResponseMetadata": {"HTTPStatusCode": 429}},
                    "InvokeModel"
                )
            return {"body": _async_iter(_standard_stream_events())}

        client.client.invoke_model_with_response_stream = mock_invoke

        events = []
        request = TextRequest(prompt="retry event test unique prompt", temperature=0, use_cache=False)
        request.on_progress = lambda e: events.append(e)

        result = await client.generate_text_streamed(request)

        # Verify retry event was fired
        retry_events = [e for e in events if e.get("event") == "retry"]
        assert len(retry_events) == 1
        assert retry_events[0]["attempt"] == 1
        assert retry_events[0]["max_retries"] == 2
        assert retry_events[0]["error"] == "ClientError"
        assert "delay" in retry_events[0]

        # Verify successful result
        assert result.text == "Hello world"


class TestAllRetriesExhausted:
    """Test all retries exhausted raises final exception.

    **Validates: Requirements 6.4**
    """

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises(self):
        """When all retries are exhausted, the final exception is raised."""
        client = _create_client(max_retries=2, retry_delay=0.001, max_retry_delay=0.01)

        from botocore.exceptions import ClientError

        client.client.invoke_model_with_response_stream = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "throttled"},
                 "ResponseMetadata": {"HTTPStatusCode": 429}},
                "InvokeModel"
            )
        )

        request = TextRequest(prompt="always fail", temperature=0)

        with pytest.raises(ClientError):
            await client.generate_text_streamed(request)

        # Verify it was called max_retries + 1 times (initial + retries)
        assert client.client.invoke_model_with_response_stream.call_count == 3


class TestNonRetryableErrorRaisesImmediately:
    """Test non-retryable error raises immediately without retry.

    **Validates: Requirements 6.6**
    """

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        """Non-retryable errors raise immediately without retry."""
        client = _create_client(max_retries=3)

        # A generic ValueError is not retryable
        client.client.invoke_model_with_response_stream.side_effect = RuntimeError("bad model")

        request = TextRequest(prompt="non-retryable", temperature=0)

        with pytest.raises(RuntimeError, match="bad model"):
            await client.generate_text_streamed(request)

        # Only called once (no retries)
        assert client.client.invoke_model_with_response_stream.call_count == 1


# --- Thinking Content Tests ---


class TestReasoningTokensFromMetadata:
    """Test reasoning_tokens populated from Bedrock metadata.

    **Validates: Requirements 3.2**
    """

    @pytest.mark.asyncio
    async def test_reasoning_tokens_from_stream_metadata(self):
        """reasoning_tokens is populated from message_delta metadata."""
        client = _create_client()

        stream_events = [
            _make_message_start_event(input_tokens=20),
            _make_thinking_delta_event("Let me think..."),
            _make_text_delta_event("The answer is 42"),
            _make_message_delta_event(output_tokens=30, reasoning_tokens=15, stop_reason="end_turn"),
            _make_message_stop_event(),
        ]
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        request = TextRequest(prompt="think about this unique reasoning test", temperature=0, use_cache=False)
        result = await client.generate_text_streamed(request)

        assert result.reasoning_tokens == 15
        assert result.metadata.get("thinking") == "Let me think..."


class TestNoThinkingContent:
    """Test no thinking content → reasoning_tokens=0, no metadata["thinking"] key.

    **Validates: Requirements 3.4**
    """

    @pytest.mark.asyncio
    async def test_no_thinking_means_zero_reasoning_tokens(self):
        """When no thinking content arrives, reasoning_tokens=0 and no thinking key."""
        client = _create_client()

        stream_events = [
            _make_message_start_event(input_tokens=10),
            _make_text_delta_event("simple answer"),
            _make_message_delta_event(output_tokens=5, reasoning_tokens=0, stop_reason="end_turn"),
            _make_message_stop_event(),
        ]
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        request = TextRequest(prompt="no thinking", temperature=0)
        result = await client.generate_text_streamed(request)

        assert result.reasoning_tokens == 0
        assert "thinking" not in result.metadata


# --- Structured Output Rejection Tests ---


class TestValueErrorMessageSuggestsGenerateText:
    """Test ValueError message suggests generate_text alternative.

    **Validates: Requirements 8.1, 8.2**
    """

    @pytest.mark.asyncio
    async def test_value_error_message(self):
        """ValueError message mentions generate_text as alternative."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        client = _create_client()

        request = TextRequest(prompt="structured", response_format=MyModel)

        with pytest.raises(ValueError) as exc_info:
            await client.generate_text_streamed(request)

        error_msg = str(exc_info.value)
        assert "generate_text" in error_msg
        assert "structured output" in error_msg.lower() or "response_format" in error_msg.lower()


# --- TextResponse Shape Tests ---


class TestTextResponseHasSameFieldsAsGenerateText:
    """Test TextResponse has same fields as generate_text response.

    **Validates: Requirements 1.3**
    """

    @pytest.mark.asyncio
    async def test_text_response_has_all_fields(self):
        """TextResponse from generate_text_streamed contains all expected fields."""
        client = _create_client()

        stream_events = [
            _make_message_start_event(input_tokens=20),
            _make_text_delta_event("generated text"),
            _make_message_delta_event(output_tokens=10, reasoning_tokens=0, stop_reason="end_turn"),
            _make_message_stop_event(),
        ]
        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(stream_events)
        }

        request = TextRequest(prompt="test fields", temperature=0)
        result = await client.generate_text_streamed(request)

        # Verify all TextResponse fields are present and have correct types
        assert isinstance(result, TextResponse)
        assert isinstance(result.text, str) and result.text == "generated text"
        assert isinstance(result.model, str)
        assert isinstance(result.stop_reason, str) and result.stop_reason == "end_turn"
        assert isinstance(result.input_tokens, int) and result.input_tokens == 20
        assert isinstance(result.output_tokens, int) and result.output_tokens == 10
        assert isinstance(result.reasoning_tokens, int)
        assert isinstance(result.cached_tokens, int)
        assert result.timestamp is not None
        assert isinstance(result.elapsed_seconds, float)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.cache_source, str)
