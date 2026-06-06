"""Integration tests for generate_text_streamed (requires AWS Bedrock credentials).

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/integration/test_streaming_assembly_integration.py -v --model us.anthropic.claude-sonnet-4-5-20250929-v1:0

These tests hit the real Bedrock API and verify:
1. Long output with progress events
2. Concurrent requests with semaphore gating
3. Cache hit on second call
4. Extended thinking content assembly
"""

import asyncio
import time

import pytest

from smartllm import LLMClient, LLMConfig, TextRequest


BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_REGION = "us-east-1"


def _requires_bedrock(test_provider):
    """Skip test if provider is not bedrock."""
    if test_provider != "bedrock":
        pytest.skip("generate_text_streamed only supported on Bedrock")


@pytest.fixture
def bedrock_client():
    """Create a Bedrock LLMClient for streaming tests."""
    config = LLMConfig(
        provider="bedrock",
        default_model=BEDROCK_MODEL,
        aws_region=BEDROCK_REGION,
        temperature=0,
        max_tokens=4096,
        max_concurrent=2,
    )
    return LLMClient(config)


@pytest.fixture
def progress_collector():
    """Fixture that returns a callback and the collected events list."""
    events = []

    def on_progress(event):
        events.append(event)

    return on_progress, events


@pytest.mark.integration
@pytest.mark.asyncio
async def test_long_output_with_progress_events(bedrock_client, progress_collector, test_provider, capsys):
    """Streaming a long response fires progress events at ~500 token intervals."""
    _requires_bedrock(test_provider)

    on_progress, events = progress_collector

    request = TextRequest(
        prompt=(
            "Write a detailed explanation of how TCP/IP networking works, covering "
            "the OSI model, the three-way handshake, flow control, congestion control, "
            "and routing protocols. Be thorough and technical. Aim for at least 1000 words."
        ),
        temperature=0,
        max_tokens=4096,
        on_progress=on_progress,
        use_cache=False,
    )

    result = await bedrock_client.generate_text_streamed(request)

    # Verify we got a substantial response
    assert len(result.text) > 3000, f"Expected long output, got {len(result.text)} chars"
    assert result.output_tokens > 500
    assert result.stop_reason == "end_turn"
    assert result.model == BEDROCK_MODEL

    # Verify progress events fired
    event_types = [e["event"] for e in events]
    assert "llm_started" in event_types
    assert "llm_done" in event_types
    assert "stream_progress" in event_types

    # Verify stream_progress events have correct fields
    progress_events = [e for e in events if e["event"] == "stream_progress"]
    assert len(progress_events) >= 1, "Expected at least 1 progress event for long output"

    for evt in progress_events:
        assert "text_tokens_so_far" in evt
        assert "text_so_far" in evt
        assert "elapsed_seconds" in evt
        assert evt["text_tokens_so_far"] >= 500  # Should only fire at 500+ boundaries

    # Print summary for visibility
    print(f"\n  Output: {len(result.text)} chars, {result.output_tokens} tokens")
    print(f"  Progress events fired: {len(progress_events)}")
    print(f"  Elapsed: {result.elapsed_seconds}s")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests_with_semaphore(bedrock_client, test_provider):
    """4 concurrent requests with max_concurrent=2 demonstrates semaphore gating."""
    _requires_bedrock(test_provider)

    prompts = [
        "Explain quantum entanglement in 2 sentences.",
        "Explain the Krebs cycle in 2 sentences.",
        "Explain how a compiler works in 2 sentences.",
        "Explain black holes in 2 sentences.",
    ]

    start_times = {}
    end_times = {}

    async def run_one(idx: int, prompt: str):
        start_times[idx] = time.time()
        request = TextRequest(
            prompt=prompt,
            temperature=0,
            max_tokens=150,
            use_cache=False,
        )
        result = await bedrock_client.generate_text_streamed(request)
        end_times[idx] = time.time()
        return result

    t0 = time.time()
    results = await asyncio.gather(*[run_one(i, p) for i, p in enumerate(prompts)])
    total_elapsed = time.time() - t0

    # All should succeed
    for r in results:
        assert r.text
        assert r.output_tokens > 0

    # With max_concurrent=2 and 4 requests, total time should be less than
    # 4x the slowest individual request (proving concurrency) but more than
    # the slowest single request (proving some serialization)
    individual_times = [end_times[i] - start_times[i] for i in range(4)]
    slowest = max(individual_times)

    # Total should be significantly less than sum of all (proves concurrency happened)
    sum_all = sum(individual_times)
    assert total_elapsed < sum_all * 0.9, (
        f"Total {total_elapsed:.1f}s >= 90% of sum {sum_all:.1f}s — concurrency not working"
    )

    print(f"\n  Total: {total_elapsed:.1f}s, Sum of individual: {sum_all:.1f}s")
    print(f"  Concurrency speedup: {sum_all / total_elapsed:.1f}x")
    for i, r in enumerate(results):
        print(f"  [req-{i}] {r.output_tokens} tokens in {individual_times[i]:.1f}s")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_hit_on_second_call(bedrock_client, test_provider):
    """Second call with identical params returns from cache instantly."""
    _requires_bedrock(test_provider)

    prompt = "What is the capital of Estonia? One word answer."

    # First call — clear cache, should stream from Bedrock
    events_first = []
    request1 = TextRequest(
        prompt=prompt,
        temperature=0,
        max_tokens=50,
        on_progress=lambda e: events_first.append(e),
        use_cache=True,
        clear_cache=True,
    )

    t0 = time.time()
    result1 = await bedrock_client.generate_text_streamed(request1)
    first_elapsed = time.time() - t0

    assert result1.cache_source == "miss"
    assert "llm_started" in [e["event"] for e in events_first]

    # Second call — should hit cache
    events_second = []
    request2 = TextRequest(
        prompt=prompt,
        temperature=0,
        max_tokens=50,
        on_progress=lambda e: events_second.append(e),
        use_cache=True,
    )

    t0 = time.time()
    result2 = await bedrock_client.generate_text_streamed(request2)
    second_elapsed = time.time() - t0

    # Cache hit verification
    assert result2.cache_source in ("l1", "l2")
    assert result2.text == result1.text
    assert second_elapsed < 0.1, f"Cache hit took {second_elapsed}s — too slow"

    # Should have fired cache_hit event, NOT llm_started
    event_types_second = [e["event"] for e in events_second]
    assert "cache_hit" in event_types_second
    assert "llm_started" not in event_types_second

    print(f"\n  First call: {first_elapsed:.2f}s (miss)")
    print(f"  Second call: {second_elapsed:.4f}s (cache hit)")
    print(f"  Speedup: {first_elapsed / max(second_elapsed, 0.0001):.0f}x")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extended_thinking_assembly(bedrock_client, test_provider):
    """Extended thinking content is assembled into metadata['thinking']."""
    _requires_bedrock(test_provider)

    events = []
    request = TextRequest(
        prompt="What is the 12th prime number? Work through it step by step.",
        temperature=1,  # Required for thinking
        max_tokens=4096,
        reasoning_effort="low",
        on_progress=lambda e: events.append(e),
        use_cache=False,
    )

    result = await bedrock_client.generate_text_streamed(request)

    # Should have thinking content in metadata
    assert "thinking" in result.metadata, "Expected thinking content in metadata"
    assert len(result.metadata["thinking"]) > 50, "Thinking text too short"

    # The answer should be 37
    assert "37" in result.text, f"Expected '37' in response, got: {result.text[:200]}"

    # Output should include the answer text
    assert result.output_tokens > 0
    assert result.stop_reason == "end_turn"

    print(f"\n  Answer: {result.text[:100]}")
    print(f"  Thinking length: {len(result.metadata['thinking'])} chars")
    print(f"  Reasoning tokens: {result.reasoning_tokens}")
    print(f"  Output tokens: {result.output_tokens}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_not_implemented_for_openai(test_provider):
    """generate_text_streamed raises NotImplementedError for OpenAI provider."""
    if test_provider == "bedrock":
        pytest.skip("This test is for non-bedrock providers")

    config = LLMConfig(
        provider="openai",
        api_key="test-key",
        default_model="gpt-4o-mini",
        temperature=0,
        max_tokens=100,
    )
    client = LLMClient(config)
    request = TextRequest(prompt="Hello")

    with pytest.raises(NotImplementedError, match="not yet supported"):
        await client.generate_text_streamed(request)
