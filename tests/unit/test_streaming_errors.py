"""Unit tests for stream-level error handling and timeouts.

Covers the three issues from `tmp/smartllm-observability-bug-report.md`:
  1. Bedrock stream-level error events are detected and raised as
     `BedrockStreamError` instead of being silently dropped.
  2. Streams that exceed `stream_total_timeout` raise
     `BedrockStreamTimeoutError(kind="total")`.
  3. Streams that don't deliver a first event within
     `stream_first_chunk_timeout` raise
     `BedrockStreamTimeoutError(kind="first_chunk")`.

The wrapper `_iter_stream_safely` is exercised through all three streaming
methods (`generate_text_streamed`, `generate_text_stream`,
`send_message_stream`) to confirm consistent behavior.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from smartllm.bedrock import (
    BedrockError,
    BedrockStreamError,
    BedrockStreamTimeoutError,
)
from smartllm.bedrock.bedrock_client import BedrockLLMClient
from smartllm.bedrock.config import BedrockConfig
from smartllm.bedrock.exceptions import (
    STREAM_ERROR_EVENT_KEYS,
    RETRYABLE_STREAM_ERROR_TYPES,
)
from smartllm.models import TextRequest, MessageRequest, Message
from smartllm.utils.retry_utils import is_retryable_error


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


def _make_message_delta_event(output_tokens: int = 5, stop_reason: str = "end_turn") -> dict:
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "message_delta",
                "usage": {"output_tokens": output_tokens, "reasoning_tokens": 0},
                "delta": {"stop_reason": stop_reason},
            }).encode()
        }
    }


def _make_message_stop_event() -> dict:
    return {"chunk": {"bytes": json.dumps({"type": "message_stop"}).encode()}}


def _standard_stream_events(text: str = "Hello") -> list:
    return [
        _make_message_start_event(),
        _make_text_delta_event(text),
        _make_message_delta_event(),
        _make_message_stop_event(),
    ]


async def _async_iter(events):
    for event in events:
        yield event


def _create_client(
    max_retries: int = 0,
    retry_delay: float = 0.01,
    max_retry_delay: float = 0.05,
    stream_total_timeout: float = 900,
    stream_first_chunk_timeout: float = 60,
) -> BedrockLLMClient:
    """Create a BedrockLLMClient for testing without real AWS connection."""
    config = BedrockConfig(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_region="us-east-1",
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_retry_delay=max_retry_delay,
        stream_total_timeout=stream_total_timeout,
        stream_first_chunk_timeout=stream_first_chunk_timeout,
    )
    client = BedrockLLMClient(config=config, max_concurrent=2)
    client.client = AsyncMock()
    return client


# ===========================================================================
# Issue 1 — Stream-level error events
# ===========================================================================


class TestStreamErrorEventsAreRaised:
    """Bedrock stream-level error events trigger BedrockStreamError instead
    of being silently dropped (see bug report Issue 1)."""

    @pytest.mark.parametrize("error_key", list(STREAM_ERROR_EVENT_KEYS))
    @pytest.mark.asyncio
    async def test_each_documented_error_event_raises(self, error_key):
        """All six documented Bedrock stream-error event types raise
        BedrockStreamError with the correct error_type."""
        client = _create_client()

        async def stream_with_error():
            yield _make_message_start_event()
            yield {error_key: {"message": f"simulated {error_key}"}}

        client.client.invoke_model_with_response_stream.return_value = {
            "body": stream_with_error()
        }

        request = TextRequest(prompt=f"err {error_key}", temperature=0, use_cache=False)

        with pytest.raises(BedrockStreamError) as exc_info:
            await client.generate_text_streamed(request)

        assert exc_info.value.error_type == error_key
        assert f"simulated {error_key}" in exc_info.value.message
        assert exc_info.value.raw == {error_key: {"message": f"simulated {error_key}"}}

    @pytest.mark.asyncio
    async def test_throttling_in_generate_text_stream_raises(self):
        """The legacy generate_text_stream method also raises on errors."""
        client = _create_client()

        async def stream_with_error():
            yield _make_message_start_event()
            yield {"throttlingException": {"message": "rate limited"}}

        client.client.invoke_model_with_response_stream.return_value = {
            "body": stream_with_error()
        }

        request = TextRequest(prompt="legacy stream", temperature=0, stream=True)

        with pytest.raises(BedrockStreamError) as exc_info:
            chunks = []
            async for chunk in client.generate_text_stream(request):
                chunks.append(chunk)

        assert exc_info.value.error_type == "throttlingException"

    @pytest.mark.asyncio
    async def test_throttling_in_send_message_stream_raises(self):
        """send_message_stream also raises on stream-level errors."""
        client = _create_client()

        async def stream_with_error():
            yield _make_message_start_event()
            yield {"modelTimeoutException": {"message": "model timed out"}}

        client.client.invoke_model_with_response_stream.return_value = {
            "body": stream_with_error()
        }

        request = MessageRequest(
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        with pytest.raises(BedrockStreamError) as exc_info:
            async for _ in client.send_message_stream(request):
                pass

        assert exc_info.value.error_type == "modelTimeoutException"

    @pytest.mark.asyncio
    async def test_unknown_event_keys_logged_not_raised(self, caplog):
        """Events with no chunk/error keys are logged at WARNING level and skipped."""
        import logging

        client = _create_client()

        async def stream_with_unknown_event():
            yield _make_message_start_event()
            yield {"someFutureEventType": {"data": "weird"}}
            yield _make_text_delta_event("ok")
            yield _make_message_delta_event()
            yield _make_message_stop_event()

        client.client.invoke_model_with_response_stream.return_value = {
            "body": stream_with_unknown_event()
        }

        request = TextRequest(prompt="unknown event", temperature=0, use_cache=False)

        with caplog.at_level(logging.WARNING, logger="smartllm"):
            result = await client.generate_text_streamed(request)

        assert result.text == "ok"
        # A warning was logged with the unknown event key
        assert any(
            "someFutureEventType" in record.getMessage() for record in caplog.records
        )


class TestStreamErrorRetryability:
    """BedrockStreamError participates in the retry loop based on error_type."""

    @pytest.mark.parametrize("error_type", list(RETRYABLE_STREAM_ERROR_TYPES))
    def test_retryable_error_types_recognized(self, error_type):
        err = BedrockStreamError(error_type=error_type, message="x")
        assert err.is_retryable is True
        assert is_retryable_error(err) is True

    def test_validation_error_not_retryable(self):
        err = BedrockStreamError(error_type="validationException", message="bad shape")
        assert err.is_retryable is False
        assert is_retryable_error(err) is False

    @pytest.mark.asyncio
    async def test_throttling_event_triggers_retry_then_success(self):
        """A throttling stream-event on attempt 1 should be retried, succeeding on attempt 2."""
        client = _create_client(max_retries=2, retry_delay=0.001, max_retry_delay=0.01)

        call_count = 0

        async def make_stream():
            return _async_iter([
                _make_message_start_event(),
                {"throttlingException": {"message": "throttled"}},
            ])

        async def make_success_stream():
            return _async_iter(_standard_stream_events("retried success"))

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"body": await make_stream()}
            return {"body": await make_success_stream()}

        client.client.invoke_model_with_response_stream = mock_invoke

        events = []
        request = TextRequest(
            prompt="retry on stream throttle",
            temperature=0,
            use_cache=False,
        )
        request.on_progress = lambda e: events.append(e)

        result = await client.generate_text_streamed(request)

        assert result.text == "retried success"
        retry_events = [e for e in events if e.get("event") == "retry"]
        assert len(retry_events) == 1
        assert retry_events[0]["error"] == "BedrockStreamError"


# ===========================================================================
# Issue 2 — Total stream timeout
# ===========================================================================


class TestStreamTotalTimeout:
    """Streams that exceed stream_total_timeout raise BedrockStreamTimeoutError."""

    @pytest.mark.asyncio
    async def test_stalling_stream_raises_total_timeout(self):
        """A stream that delivers one event then stalls forever should
        raise BedrockStreamTimeoutError(kind='total') after the budget."""
        client = _create_client(
            stream_total_timeout=0.2,
            stream_first_chunk_timeout=0.5,  # > total, so total fires first
        )

        async def stalling_stream():
            yield _make_message_start_event()
            await asyncio.sleep(5)  # never completes within timeout
            yield _make_message_stop_event()

        client.client.invoke_model_with_response_stream.return_value = {
            "body": stalling_stream()
        }

        request = TextRequest(prompt="stall", temperature=0, use_cache=False)

        with pytest.raises(BedrockStreamTimeoutError) as exc_info:
            await client.generate_text_streamed(request)

        assert exc_info.value.kind == "total"
        assert exc_info.value.elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_total_timeout_disabled_by_zero(self):
        """stream_total_timeout=0 disables the total timeout."""
        client = _create_client(
            stream_total_timeout=0,  # disabled
            stream_first_chunk_timeout=0,  # disabled
        )

        client.client.invoke_model_with_response_stream.return_value = {
            "body": _async_iter(_standard_stream_events("ok"))
        }

        request = TextRequest(prompt="no timeout", temperature=0, use_cache=False)
        result = await client.generate_text_streamed(request)
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_total_timeout_in_generate_text_stream(self):
        """Total timeout also fires from the iterator-returning streaming method."""
        client = _create_client(
            stream_total_timeout=0.2,
            stream_first_chunk_timeout=0.5,
        )

        async def stalling_stream():
            yield _make_message_start_event()
            yield _make_text_delta_event("partial")
            await asyncio.sleep(5)
            yield _make_message_stop_event()

        client.client.invoke_model_with_response_stream.return_value = {
            "body": stalling_stream()
        }

        request = TextRequest(prompt="stall iter", temperature=0, stream=True)

        with pytest.raises(BedrockStreamTimeoutError) as exc_info:
            async for _ in client.generate_text_stream(request):
                pass

        assert exc_info.value.kind == "total"


# ===========================================================================
# Issue 3 — First-chunk timeout
# ===========================================================================


class TestStreamFirstChunkTimeout:
    """Streams that don't deliver a first event within
    stream_first_chunk_timeout raise BedrockStreamTimeoutError."""

    @pytest.mark.asyncio
    async def test_delayed_first_event_raises_first_chunk_timeout(self):
        """A stream that doesn't deliver any event within the first-chunk
        budget should raise BedrockStreamTimeoutError(kind='first_chunk')."""
        client = _create_client(
            stream_first_chunk_timeout=0.1,
            stream_total_timeout=10,  # large enough that first_chunk fires first
        )

        async def slow_first_stream():
            await asyncio.sleep(2)  # never delivers within first-chunk budget
            yield _make_message_start_event()

        client.client.invoke_model_with_response_stream.return_value = {
            "body": slow_first_stream()
        }

        request = TextRequest(prompt="slow first", temperature=0, use_cache=False)

        with pytest.raises(BedrockStreamTimeoutError) as exc_info:
            await client.generate_text_streamed(request)

        assert exc_info.value.kind == "first_chunk"
        assert exc_info.value.elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_fast_first_event_completes_normally(self):
        """A stream that delivers the first event quickly is unaffected by
        the first-chunk timeout, even if total elapsed is much longer."""
        client = _create_client(
            stream_first_chunk_timeout=0.1,
            stream_total_timeout=10,
        )

        async def quick_first_then_pause():
            yield _make_message_start_event()
            yield _make_text_delta_event("fast")
            # No further delays — completes promptly
            yield _make_message_delta_event()
            yield _make_message_stop_event()

        client.client.invoke_model_with_response_stream.return_value = {
            "body": quick_first_then_pause()
        }

        request = TextRequest(prompt="fast first", temperature=0, use_cache=False)
        result = await client.generate_text_streamed(request)
        assert result.text == "fast"

    @pytest.mark.asyncio
    async def test_first_chunk_timeout_disabled_by_zero(self):
        """stream_first_chunk_timeout=0 disables the first-chunk guard."""
        client = _create_client(
            stream_first_chunk_timeout=0,
            stream_total_timeout=5,
        )

        async def slow_first_stream():
            await asyncio.sleep(0.3)  # would trip a normal first-chunk timeout
            for ev in _standard_stream_events("ok"):
                yield ev

        client.client.invoke_model_with_response_stream.return_value = {
            "body": slow_first_stream()
        }

        request = TextRequest(prompt="no fc timeout", temperature=0, use_cache=False)
        result = await client.generate_text_streamed(request)
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_first_chunk_timeout_in_send_message_stream(self):
        """First-chunk timeout also fires from send_message_stream."""
        client = _create_client(
            stream_first_chunk_timeout=0.1,
            stream_total_timeout=10,
        )

        async def slow_first_stream():
            await asyncio.sleep(2)
            yield _make_message_start_event()

        client.client.invoke_model_with_response_stream.return_value = {
            "body": slow_first_stream()
        }

        request = MessageRequest(
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        with pytest.raises(BedrockStreamTimeoutError) as exc_info:
            async for _ in client.send_message_stream(request):
                pass

        assert exc_info.value.kind == "first_chunk"


# ===========================================================================
# Exception hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    """Bedrock-specific exceptions inherit from a common base so consumers
    can catch broadly with one `except`."""

    def test_stream_error_inherits_from_bedrock_error(self):
        err = BedrockStreamError(error_type="throttlingException")
        assert isinstance(err, BedrockError)
        assert isinstance(err, Exception)

    def test_timeout_error_inherits_from_bedrock_error(self):
        err = BedrockStreamTimeoutError(kind="total", elapsed=10.0)
        assert isinstance(err, BedrockError)

    def test_timeout_error_validates_kind(self):
        with pytest.raises(ValueError):
            BedrockStreamTimeoutError(kind="invalid", elapsed=1.0)

    def test_timeout_error_not_retryable_by_default(self):
        err = BedrockStreamTimeoutError(kind="first_chunk", elapsed=60.0)
        assert err.is_retryable is False
        assert is_retryable_error(err) is False
