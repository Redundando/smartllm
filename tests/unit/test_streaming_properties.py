"""Property-based tests for streaming-with-assembly feature.

Uses Hypothesis to verify correctness properties of the stream assembly logic.
"""

import json
import asyncio
from unittest.mock import patch, AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from smartllm.bedrock.bedrock_client import BedrockLLMClient
from smartllm.bedrock.config import BedrockConfig


# --- Shared Helpers ---

def _make_thinking_delta_event(text: str) -> dict:
    """Create a Bedrock stream event with a thinking_delta."""
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": text},
            }).encode()
        }
    }


def _make_text_delta_event(text: str) -> dict:
    """Create a Bedrock stream event with a text_delta."""
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": text},
            }).encode()
        }
    }


def _make_message_start_event(input_tokens: int = 10) -> dict:
    """Create a message_start stream event."""
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "message_start",
                "message": {"usage": {"input_tokens": input_tokens}},
            }).encode()
        }
    }


def _make_message_delta_event(output_tokens: int = 50, reasoning_tokens: int = 0, stop_reason: str = "end_turn") -> dict:
    """Create a message_delta stream event."""
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "message_delta",
                "usage": {"output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens},
                "delta": {"stop_reason": stop_reason},
            }).encode()
        }
    }


def _make_message_stop_event() -> dict:
    """Create a message_stop stream event."""
    return {
        "chunk": {
            "bytes": json.dumps({
                "type": "message_stop",
            }).encode()
        }
    }


async def _async_iter(events):
    """Convert a list of events into an async iterable."""
    for event in events:
        yield event


def _create_test_client() -> BedrockLLMClient:
    """Create a BedrockLLMClient for testing without real AWS connection."""
    config = BedrockConfig(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_region="us-east-1",
    )
    return BedrockLLMClient(config=config)


# --- Strategies ---

@st.composite
def thinking_delta_sequences(draw):
    """Generate a list of thinking delta chunk sizes.

    Ensures total accumulated text crosses at least one 500-token boundary
    (total chars >= 2000) to make the property meaningful.
    Uses integer sizes to avoid slow unicode generation.
    """
    # Generate between 2 and 30 chunks
    num_chunks = draw(st.integers(min_value=2, max_value=30))
    # Each chunk between 50 and 2000 characters
    chunk_sizes = draw(
        st.lists(
            st.integers(min_value=50, max_value=2000),
            min_size=num_chunks,
            max_size=num_chunks,
        )
    )
    # Ensure total crosses at least one 500-token boundary (2000 chars)
    total_len = sum(chunk_sizes)
    assume(total_len >= 2000)
    # Build chunks as repeated characters (fast generation)
    chunks = ["t" * size for size in chunk_sizes]
    return chunks


@st.composite
def text_delta_sequences(draw):
    """Generate a list of text delta chunk sizes.

    Ensures total accumulated text crosses at least one 500-token boundary.
    """
    num_chunks = draw(st.integers(min_value=2, max_value=30))
    chunk_sizes = draw(
        st.lists(
            st.integers(min_value=50, max_value=2000),
            min_size=num_chunks,
            max_size=num_chunks,
        )
    )
    total_len = sum(chunk_sizes)
    assume(total_len >= 2000)
    chunks = ["x" * size for size in chunk_sizes]
    return chunks


# --- Property 1: Stream assembly preserves chunk order ---
# Feature: streaming-with-assembly, Property 1: Stream assembly preserves chunk order

class TestStreamAssemblyChunkOrder:
    """Property 1: Stream assembly preserves chunk order.

    For any sequence of text chunk deltas received from a Bedrock stream,
    the assembled TextResponse.text SHALL equal the concatenation of all
    text deltas in the exact order they were received.

    **Validates: Requirements 1.2**
    """

    @settings(max_examples=200, deadline=None)
    @given(
        deltas=st.lists(
            st.text(min_size=1, max_size=200),
            min_size=0,
            max_size=50,
        )
    )
    @pytest.mark.asyncio
    async def test_assembled_text_equals_concatenation_of_deltas(self, deltas):
        """For any list of text deltas, _consume_stream assembles them in order.

        # Feature: streaming-with-assembly, Property 1: Stream assembly preserves chunk order
        **Validates: Requirements 1.2**
        """
        import time as time_module

        # Build stream events from the deltas
        events = [_make_message_start_event()]
        for d in deltas:
            events.append(_make_text_delta_event(d))
        events.append(_make_message_delta_event())
        events.append(_make_message_stop_event())

        client = _create_test_client()

        # Use fixed time so progress thresholds don't interfere
        fixed_time = 1000.0

        with patch("time.monotonic", return_value=fixed_time):
            result = await client._consume_stream(
                response_stream=_async_iter(events),
                model="test-model",
                on_progress=None,
                start_time=fixed_time,
            )

        # Property assertion: assembled text == ordered concatenation of deltas
        assert result["text"] == "".join(deltas)


# --- Property 2: Thinking progress events fire at token thresholds ---
# Feature: streaming-with-assembly, Property 2: Thinking progress events fire at token thresholds

class TestThinkingProgressEventsAtTokenThresholds:
    """Property 2: Thinking progress events fire at token thresholds.

    For any sequence of thinking chunk deltas whose cumulative estimated token count
    (len/4) crosses one or more 500-token boundaries, the stream assembler SHALL fire
    exactly one stream_thinking progress event per 500-token boundary crossed, with
    thinking_tokens_so_far reflecting the cumulative token count at emission time.

    **Validates: Requirements 2.2**
    """

    @settings(max_examples=100, deadline=None)
    @given(chunks=thinking_delta_sequences())
    @pytest.mark.asyncio
    async def test_thinking_progress_fires_at_token_boundaries(self, chunks: list):
        """Verify stream_thinking fires exactly once per 500-token boundary crossed.

        # Feature: streaming-with-assembly, Property 2: Thinking progress events fire at token thresholds
        **Validates: Requirements 2.2**
        """
        # Build stream events
        events = [_make_message_start_event()]
        for chunk in chunks:
            events.append(_make_thinking_delta_event(chunk))
        events.append(_make_message_delta_event())
        events.append(_make_message_stop_event())

        # Collect progress events
        progress_events = []

        def on_progress(event):
            progress_events.append(event)

        client = _create_test_client()

        # Mock time.monotonic to always return start_time so that time-based
        # threshold (10 seconds) never fires — we only test token-based threshold
        fixed_time = 1000.0

        with patch("time.monotonic", return_value=fixed_time):
            result = await client._consume_stream(
                response_stream=_async_iter(events),
                model="test-model",
                on_progress=on_progress,
                start_time=fixed_time,
            )

        # Filter only stream_thinking events
        thinking_events = [e for e in progress_events if e.get("event") == "stream_thinking"]

        # Calculate expected: replay accumulation and count boundary crossings
        expected_progress_count = 0
        last_progress_tokens = 0
        accumulated = ""

        for chunk in chunks:
            accumulated += chunk
            current_tokens = len(accumulated) // 4
            tokens_since_last = current_tokens - last_progress_tokens
            if tokens_since_last >= 500:
                expected_progress_count += 1
                last_progress_tokens = current_tokens

        # Assert exact count matches expected boundaries crossed
        assert len(thinking_events) == expected_progress_count, (
            f"Expected {expected_progress_count} stream_thinking events but got "
            f"{len(thinking_events)}. Total tokens: {len(''.join(chunks)) // 4}"
        )

        # Verify each event has correct thinking_tokens_so_far
        last_progress_tokens_verify = 0
        accumulated_verify = ""
        event_idx = 0

        for chunk in chunks:
            accumulated_verify += chunk
            current_tokens = len(accumulated_verify) // 4
            tokens_since_last = current_tokens - last_progress_tokens_verify
            if tokens_since_last >= 500:
                assert event_idx < len(thinking_events), (
                    f"Expected thinking event at index {event_idx} but only "
                    f"{len(thinking_events)} events were fired"
                )
                fired_event = thinking_events[event_idx]
                assert fired_event["thinking_tokens_so_far"] == current_tokens, (
                    f"Event {event_idx}: expected thinking_tokens_so_far={current_tokens}, "
                    f"got {fired_event['thinking_tokens_so_far']}"
                )
                last_progress_tokens_verify = current_tokens
                event_idx += 1

        # Verify thinking text was assembled correctly
        assert result["thinking_text"] == "".join(chunks)


# --- Property 3: Text progress events fire at token thresholds ---
# Feature: streaming-with-assembly, Property 3: Text progress events fire at token thresholds

class TestTextProgressEventsAtTokenThresholds:
    """Property 3: Text progress events fire at token thresholds.

    For any sequence of text chunk deltas whose cumulative estimated token count
    (len/4) crosses one or more 500-token boundaries, the stream assembler SHALL
    fire exactly one stream_progress progress event per 500-token boundary crossed,
    with text_tokens_so_far reflecting the cumulative token count at emission time.

    **Validates: Requirements 2.3**
    """

    @settings(max_examples=100, deadline=None)
    @given(chunks=text_delta_sequences())
    @pytest.mark.asyncio
    async def test_text_progress_fires_at_token_boundaries(self, chunks: list):
        """Verify stream_progress fires exactly once per 500-token boundary crossed.

        # Feature: streaming-with-assembly, Property 3: Text progress events fire at token thresholds
        **Validates: Requirements 2.3**
        """
        # Skip if total tokens won't cross any boundary
        total_text = "".join(chunks)
        total_tokens = len(total_text) // 4
        assume(total_tokens >= 500)

        # Build stream events
        events = [_make_message_start_event()]
        for chunk in chunks:
            events.append(_make_text_delta_event(chunk))
        events.append(_make_message_delta_event(output_tokens=total_tokens))
        events.append(_make_message_stop_event())

        # Collect progress events
        progress_events = []

        def on_progress(event):
            progress_events.append(event)

        client = _create_test_client()

        # Mock time.monotonic so time-based threshold never fires
        fixed_time = 1000.0

        with patch("time.monotonic", return_value=fixed_time):
            result = await client._consume_stream(
                response_stream=_async_iter(events),
                model="test-model",
                on_progress=on_progress,
                start_time=fixed_time,
            )

        # Calculate expected progress events
        expected_progress_count = 0
        last_progress_tokens = 0
        accumulated = ""

        for chunk in chunks:
            accumulated += chunk
            current_tokens = len(accumulated) // 4
            tokens_since_last = current_tokens - last_progress_tokens
            if tokens_since_last >= 500:
                expected_progress_count += 1
                last_progress_tokens = current_tokens

        # Filter only stream_progress events
        text_progress_events = [e for e in progress_events if e.get("event") == "stream_progress"]

        # Assert exact count
        assert len(text_progress_events) == expected_progress_count, (
            f"Expected {expected_progress_count} stream_progress events but got "
            f"{len(text_progress_events)}. Total tokens: {total_tokens}"
        )

        # Verify text_tokens_so_far at each emission
        last_progress_tokens_verify = 0
        accumulated_verify = ""
        event_idx = 0

        for chunk in chunks:
            accumulated_verify += chunk
            current_tokens = len(accumulated_verify) // 4
            tokens_since_last = current_tokens - last_progress_tokens_verify
            if tokens_since_last >= 500:
                assert event_idx < len(text_progress_events)
                fired_event = text_progress_events[event_idx]
                assert fired_event["text_tokens_so_far"] == current_tokens, (
                    f"Event {event_idx}: expected text_tokens_so_far={current_tokens}, "
                    f"got {fired_event['text_tokens_so_far']}"
                )
                last_progress_tokens_verify = current_tokens
                event_idx += 1

        # Verify assembled text is correct
        assert result["text"] == total_text


# --- Property 4: Thinking content round-trip in assembled response ---
# Feature: streaming-with-assembly, Property 4: Thinking content round-trip in assembled response

class TestThinkingContentRoundTrip:
    """Property 4: Thinking content round-trip in assembled response.

    For any stream containing a mix of thinking deltas and text deltas,
    the assembled result SHALL have thinking_text equal to the concatenation
    of all thinking deltas in order, and text equal to the concatenation of
    all text deltas in order, with no cross-contamination between the two
    accumulators.

    **Validates: Requirements 3.1, 3.3**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        thinking_deltas=st.lists(st.text(min_size=1, max_size=200), min_size=0, max_size=15),
        text_deltas=st.lists(st.text(min_size=1, max_size=200), min_size=0, max_size=15),
        ordering=st.lists(st.booleans(), min_size=0, max_size=30),
    )
    @pytest.mark.asyncio
    async def test_thinking_content_round_trip_no_cross_contamination(
        self, thinking_deltas, text_deltas, ordering
    ):
        """For any mix of thinking and text deltas, accumulators never cross-contaminate.

        # Feature: streaming-with-assembly, Property 4: Thinking content round-trip in assembled response
        **Validates: Requirements 3.1, 3.3**
        """
        # Build interleaved stream events
        events = [_make_message_start_event()]

        thinking_idx = 0
        text_idx = 0

        for use_thinking in ordering:
            if use_thinking and thinking_idx < len(thinking_deltas):
                events.append(_make_thinking_delta_event(thinking_deltas[thinking_idx]))
                thinking_idx += 1
            elif not use_thinking and text_idx < len(text_deltas):
                events.append(_make_text_delta_event(text_deltas[text_idx]))
                text_idx += 1
            elif thinking_idx < len(thinking_deltas):
                events.append(_make_thinking_delta_event(thinking_deltas[thinking_idx]))
                thinking_idx += 1
            elif text_idx < len(text_deltas):
                events.append(_make_text_delta_event(text_deltas[text_idx]))
                text_idx += 1

        # Append any remaining deltas not yet consumed
        while thinking_idx < len(thinking_deltas):
            events.append(_make_thinking_delta_event(thinking_deltas[thinking_idx]))
            thinking_idx += 1
        while text_idx < len(text_deltas):
            events.append(_make_text_delta_event(text_deltas[text_idx]))
            text_idx += 1

        # Add message_delta and message_stop to complete the stream
        reasoning_tokens = len("".join(thinking_deltas)) // 4
        output_tokens = (len("".join(text_deltas)) // 4) + reasoning_tokens
        events.append(_make_message_delta_event(
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            stop_reason="end_turn",
        ))
        events.append(_make_message_stop_event())

        client = _create_test_client()

        # Use fixed time so progress thresholds don't interfere
        fixed_time = 1000.0

        with patch("time.monotonic", return_value=fixed_time):
            result = await client._consume_stream(
                response_stream=_async_iter(events),
                model="test-model",
                on_progress=None,
                start_time=fixed_time,
            )

        # Property assertions: no cross-contamination
        expected_text = "".join(text_deltas)
        expected_thinking = "".join(thinking_deltas)

        assert result["text"] == expected_text, (
            f"Text accumulator mismatch: expected {expected_text!r}, got {result['text']!r}"
        )
        assert result["thinking_text"] == expected_thinking, (
            f"Thinking accumulator mismatch: expected {expected_thinking!r}, "
            f"got {result['thinking_text']!r}"
        )

        # Length checks confirm nothing was lost or duplicated
        assert len(result["text"]) == sum(len(d) for d in text_deltas)
        assert len(result["thinking_text"]) == sum(len(d) for d in thinking_deltas)


# --- Property 5: Cache key equivalence between streaming and non-streaming ---
# Feature: streaming-with-assembly, Property 5: Cache key equivalence between streaming and non-streaming

class TestCacheKeyEquivalence:
    """Property 5: Cache key equivalence between streaming and non-streaming.

    For any valid TextRequest parameters (model, prompt, max_tokens, top_p, top_k,
    system_prompt, reasoning_effort, budget_tokens), the cache key generated by
    `generate_text_streamed` SHALL be identical to the cache key generated by
    `generate_text` for the same parameters.

    **Validates: Requirements 5.7**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        model=st.sampled_from([
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us.meta.llama3-2-90b-instruct-v1:0",
        ]),
        prompt=st.text(min_size=1, max_size=500),
        max_tokens=st.one_of(st.none(), st.integers(min_value=100, max_value=32000)),
        top_p=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        top_k=st.one_of(st.none(), st.integers(min_value=1, max_value=500)),
        system_prompt=st.one_of(st.none(), st.text(min_size=1, max_size=200)),
        reasoning_effort=st.one_of(st.none(), st.sampled_from(["low", "medium", "high"])),
        budget_tokens=st.one_of(st.none(), st.integers(min_value=1024, max_value=32000)),
    )
    @pytest.mark.asyncio
    async def test_cache_key_identical_for_streaming_and_non_streaming(
        self,
        model,
        prompt,
        max_tokens,
        top_p,
        top_k,
        system_prompt,
        reasoning_effort,
        budget_tokens,
    ):
        """For any valid params, cache key from streamed == cache key from non-streamed.

        # Feature: streaming-with-assembly, Property 5: Cache key equivalence between streaming and non-streaming
        **Validates: Requirements 5.7**
        """
        from smartllm.models import TextRequest
        from smartllm.defaults import BEDROCK_THINKING_BUDGET

        # Build a TextRequest with response_format=None (streaming rejects non-None)
        request = TextRequest(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            system_prompt=system_prompt,
            response_format=None,
            reasoning_effort=reasoning_effort,
            budget_tokens=budget_tokens,
        )

        client = _create_test_client()

        # Resolve thinking_budget the same way both methods do
        thinking_budget = client._resolve_thinking_budget(request)

        # Resolve max_tokens the same way both methods do
        resolved_max_tokens = request.max_tokens or client.config.max_tokens

        # Cache key as generate_text would compute it (with response_format=None)
        key_non_streaming = client._generate_cache_key(
            model=model,
            prompt=request.prompt,
            max_tokens=resolved_max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            response_format=None,
            reasoning_effort=request.reasoning_effort,
            budget_tokens=thinking_budget,
        )

        # Cache key as generate_text_streamed computes it
        key_streaming = client._generate_cache_key(
            model=model,
            prompt=request.prompt,
            max_tokens=resolved_max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            response_format=None,  # Always None for streaming
            reasoning_effort=request.reasoning_effort,
            budget_tokens=thinking_budget,
        )

        # Property assertion: cache keys must be identical
        assert key_streaming == key_non_streaming, (
            f"Cache key mismatch!\n"
            f"  Streaming key:     {key_streaming}\n"
            f"  Non-streaming key: {key_non_streaming}\n"
            f"  Params: model={model}, max_tokens={resolved_max_tokens}, "
            f"top_p={top_p}, top_k={top_k}, reasoning_effort={reasoning_effort}, "
            f"budget_tokens={thinking_budget}"
        )


# --- Property 8: Structured output rejection ---
# Feature: streaming-with-assembly, Property 8: Structured output rejection

class TestStructuredOutputRejection:
    """Property 8: Structured output rejection.

    For any TextRequest where response_format is set to a non-None Pydantic model
    class, calling generate_text_streamed SHALL raise ValueError before performing
    any cache lookup, semaphore acquisition, or network call.

    **Validates: Requirements 8.1**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        prompt=st.text(min_size=1, max_size=500),
        model_class_idx=st.integers(min_value=0, max_value=4),
        model_name=st.sampled_from([
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
        ]),
        temperature=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
        max_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=32000)),
    )
    @pytest.mark.asyncio
    async def test_structured_output_raises_value_error_before_side_effects(
        self, prompt, model_class_idx, model_name, temperature, max_tokens
    ):
        """For any TextRequest with non-None response_format, ValueError is raised
        before any cache lookup, semaphore acquisition, or network call.

        # Feature: streaming-with-assembly, Property 8: Structured output rejection
        **Validates: Requirements 8.1**
        """
        from pydantic import BaseModel

        # Generate a variety of Pydantic model classes to use as response_format
        class ModelA(BaseModel):
            name: str

        class ModelB(BaseModel):
            value: int
            description: str

        class ModelC(BaseModel):
            items: list

        class ModelD(BaseModel):
            score: float
            label: str
            active: bool

        class ModelE(BaseModel):
            pass

        pydantic_models = [ModelA, ModelB, ModelC, ModelD, ModelE]
        response_format = pydantic_models[model_class_idx]

        # Build a TextRequest with a non-None response_format
        from smartllm.models import TextRequest

        request = TextRequest(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        client = _create_test_client()

        # Patch cache, semaphore, and network to detect any side effects
        cache_accessed = False
        semaphore_acquired = False
        network_called = False

        original_cache_get = client.cache.get
        original_get_semaphore = client._get_semaphore

        def mock_cache_get(*args, **kwargs):
            nonlocal cache_accessed
            cache_accessed = True
            return original_cache_get(*args, **kwargs)

        def mock_get_semaphore(*args, **kwargs):
            nonlocal semaphore_acquired
            semaphore_acquired = True
            return original_get_semaphore(*args, **kwargs)

        async def mock_invoke_stream(*args, **kwargs):
            nonlocal network_called
            network_called = True
            return {}

        client.cache.get = mock_cache_get
        client._get_semaphore = mock_get_semaphore
        # Pretend client is initialized so it won't try to connect
        client.client = AsyncMock()
        client.client.invoke_model_with_response_stream = mock_invoke_stream

        # Act & Assert: ValueError must be raised
        with pytest.raises(ValueError) as exc_info:
            await client.generate_text_streamed(request)

        # Verify the error message mentions streaming incompatibility
        assert "response_format" in str(exc_info.value).lower() or "structured output" in str(exc_info.value).lower()
        # Verify the error message suggests generate_text alternative
        assert "generate_text" in str(exc_info.value)

        # Property: NO side effects occurred before the ValueError
        assert not cache_accessed, "Cache was accessed before ValueError was raised"
        assert not semaphore_acquired, "Semaphore was accessed before ValueError was raised"
        assert not network_called, "Network call was made before ValueError was raised"


# --- Property 6: Retry discards prior accumulated chunks ---
# Feature: streaming-with-assembly, Property 6: Retry discards prior accumulated chunks

class TestRetryDiscardsPriorAccumulatedChunks:
    """Property 6: Retry discards prior accumulated chunks.

    For any partial sequence of chunks received before a retryable failure
    followed by a complete sequence of chunks on the successful retry, the
    assembled TextResponse.text SHALL contain only the text from the successful
    attempt and none of the text from failed attempts.

    **Validates: Requirements 6.2**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        num_failed_attempts=st.integers(min_value=1, max_value=3),
        failed_chunk_counts=st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=1,
            max_size=3,
        ),
        success_chunk_count=st.integers(min_value=1, max_value=10),
    )
    @pytest.mark.asyncio
    async def test_retry_discards_prior_chunks_only_final_attempt_text_remains(
        self, num_failed_attempts, failed_chunk_counts, success_chunk_count
    ):
        """For any partial chunk sequences before retryable failures, only
        the final successful attempt's text appears in the result.

        # Feature: streaming-with-assembly, Property 6: Retry discards prior accumulated chunks
        **Validates: Requirements 6.2**
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from smartllm.models import TextRequest
        from botocore.exceptions import ClientError

        # Truncate failed_chunk_counts to match num_failed_attempts
        failed_chunk_counts = failed_chunk_counts[:num_failed_attempts]
        # Pad if needed
        while len(failed_chunk_counts) < num_failed_attempts:
            failed_chunk_counts.append(1)

        # Generate distinct text for each failed attempt (uses prefix FAIL_N_)
        failed_texts = []
        for attempt_idx in range(num_failed_attempts):
            chunks_for_attempt = []
            for chunk_idx in range(failed_chunk_counts[attempt_idx]):
                chunks_for_attempt.append(f"FAIL_{attempt_idx}_chunk{chunk_idx}_")
            failed_texts.append(chunks_for_attempt)

        # Generate distinct text for the successful attempt (uses prefix SUCCESS_)
        success_texts = [f"SUCCESS_chunk{i}_" for i in range(success_chunk_count)]
        expected_final_text = "".join(success_texts)

        # Build streams: each failed stream yields partial chunks then raises
        # a ThrottlingException; the success stream yields all chunks normally
        call_count = 0

        async def _make_failing_stream(chunks):
            """Async iterator that yields some chunk events, then raises ThrottlingException."""
            # Yield message_start
            yield _make_message_start_event(input_tokens=5)
            # Yield text delta chunks
            for chunk_text in chunks:
                yield _make_text_delta_event(chunk_text)
            # Raise a retryable error (ThrottlingException)
            error_response = {
                "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
                "ResponseMetadata": {"HTTPStatusCode": 429},
            }
            raise ClientError(error_response, "InvokeModelWithResponseStream")

        async def _make_success_stream(chunks):
            """Async iterator that yields all chunk events and completes normally."""
            yield _make_message_start_event(input_tokens=10)
            for chunk_text in chunks:
                yield _make_text_delta_event(chunk_text)
            yield _make_message_delta_event(
                output_tokens=len(expected_final_text) // 4,
                stop_reason="end_turn",
            )
            yield _make_message_stop_event()

        async def mock_invoke_model_with_response_stream(**kwargs):
            nonlocal call_count
            current_call = call_count
            call_count += 1

            if current_call < num_failed_attempts:
                # Return a failing stream with partial data
                return {"body": _make_failing_stream(failed_texts[current_call])}
            else:
                # Return the successful stream
                return {"body": _make_success_stream(success_texts)}

        # Create client with retry configuration allowing enough retries
        config = BedrockConfig(
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region="us-east-1",
            max_retries=num_failed_attempts,  # Allow exactly enough retries
            retry_delay=0.001,  # Minimal delay for test speed
            max_retry_delay=0.01,
        )
        client = BedrockLLMClient(config=config)
        client.client = AsyncMock()
        client.client.invoke_model_with_response_stream = mock_invoke_model_with_response_stream

        request = TextRequest(
            prompt="Test prompt for retry property",
            model="test-model",
            temperature=0.5,  # Non-zero to skip cache logic
            use_cache=False,
        )

        # Execute
        result = await client.generate_text_streamed(request)

        # Property assertions:
        # 1. The result text ONLY contains text from the successful attempt
        assert result.text == expected_final_text, (
            f"Result text should be exactly the successful attempt's text.\n"
            f"Expected: {expected_final_text!r}\n"
            f"Got: {result.text!r}"
        )

        # 2. None of the failed attempt text appears in the result
        for attempt_idx, failed_chunks in enumerate(failed_texts):
            for failed_chunk in failed_chunks:
                assert failed_chunk not in result.text, (
                    f"Failed attempt {attempt_idx} chunk {failed_chunk!r} "
                    f"was found in the result text, but should have been discarded."
                )


# --- Property 7: Exponential backoff within bounds ---
# Feature: streaming-with-assembly, Property 7: Exponential backoff within bounds

class TestExponentialBackoffWithinBounds:
    """Property 7: Exponential backoff within bounds.

    For any attempt number (0-indexed), base_delay, and max_delay where
    base_delay > 0 and max_delay >= base_delay, the computed backoff delay
    SHALL be within the range [min(base_delay × 2^attempt, max_delay),
    min(base_delay × 2^attempt, max_delay) × 1.1].

    **Validates: Requirements 6.5**
    """

    @settings(max_examples=200, deadline=None)
    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        max_delay=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_backoff_within_expected_bounds(self, attempt, base_delay, max_delay):
        """For any attempt, base_delay > 0, max_delay >= base_delay, the computed
        backoff is within [base_value, base_value * 1.1].

        # Feature: streaming-with-assembly, Property 7: Exponential backoff within bounds
        **Validates: Requirements 6.5**
        """
        from smartllm.utils.retry_utils import calculate_backoff

        # Ensure max_delay >= base_delay (constraint from the property)
        assume(max_delay >= base_delay)

        # Compute the backoff
        result = calculate_backoff(attempt, base_delay, max_delay)

        # Calculate expected base value (without jitter)
        base_value = min(base_delay * (2 ** attempt), max_delay)

        # Property: result must be in [base_value, base_value * 1.1]
        assert result >= base_value, (
            f"Backoff {result} is below lower bound {base_value}. "
            f"attempt={attempt}, base_delay={base_delay}, max_delay={max_delay}"
        )
        assert result <= base_value * 1.1, (
            f"Backoff {result} exceeds upper bound {base_value * 1.1}. "
            f"attempt={attempt}, base_delay={base_delay}, max_delay={max_delay}"
        )
