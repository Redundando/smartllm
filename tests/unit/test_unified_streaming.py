"""Unit tests for LLMClient.generate_text_streamed delegation (Task 5.2)

Tests:
- Bedrock provider delegates correctly to self._client.generate_text_streamed(request)
- OpenAI provider raises NotImplementedError
- ValueError propagation from response_format

Validates: Requirements 1.4, 1.5, 8.3
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import BaseModel

from smartllm import LLMClient, LLMConfig, TextRequest, TextResponse


BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class DummySchema(BaseModel):
    """Dummy Pydantic model for structured output rejection tests."""
    answer: str


@pytest.fixture
def bedrock_config():
    """LLMConfig for Bedrock provider."""
    return LLMConfig(provider="bedrock", default_model=BEDROCK_MODEL, temperature=0, max_tokens=100)


@pytest.fixture
def openai_config():
    """LLMConfig for OpenAI provider."""
    return LLMConfig(provider="openai", api_key="test-key", default_model="gpt-4o-mini", temperature=0, max_tokens=100)


@pytest.mark.asyncio
async def test_bedrock_provider_delegates_to_client(bedrock_config):
    """Test that LLMClient.generate_text_streamed delegates to self._client.generate_text_streamed for Bedrock."""
    client = LLMClient(bedrock_config)

    expected_response = TextResponse(
        text="Hello world",
        model=BEDROCK_MODEL,
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        cached_tokens=0,
        timestamp="2024-01-01T00:00:00",
        elapsed_seconds=1.5,
        metadata={},
        cache_source="miss",
    )

    request = TextRequest(prompt="Say hello", temperature=0)

    with patch.object(client._client, 'generate_text_streamed', new_callable=AsyncMock) as mock_streamed:
        mock_streamed.return_value = expected_response
        result = await client.generate_text_streamed(request)

        # Verify delegation happened with the exact request
        mock_streamed.assert_called_once_with(request)
        # Verify the response is passed through unchanged
        assert result is expected_response
        assert result.text == "Hello world"
        assert result.model == BEDROCK_MODEL
        assert result.input_tokens == 10
        assert result.output_tokens == 5


@pytest.mark.asyncio
async def test_openai_provider_raises_not_implemented(openai_config):
    """Test that LLMClient.generate_text_streamed raises NotImplementedError for OpenAI."""
    client = LLMClient(openai_config)
    request = TextRequest(prompt="Say hello")

    with pytest.raises(NotImplementedError) as exc_info:
        await client.generate_text_streamed(request)

    # Verify the error message mentions OpenAI and suggests alternatives
    assert "OpenAI" in str(exc_info.value)
    assert "not yet supported" in str(exc_info.value).lower() or "not supported" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_openai_not_implemented_does_not_call_provider(openai_config):
    """Test that OpenAI provider's underlying client is never called for streaming."""
    client = LLMClient(openai_config)
    request = TextRequest(prompt="Say hello")

    # Patch generate_text to ensure no fallback to non-streaming method occurs
    with patch.object(client._client, 'generate_text', new_callable=AsyncMock) as mock_generate:
        with pytest.raises(NotImplementedError):
            await client.generate_text_streamed(request)

        # No method on the underlying OpenAI client should be called
        mock_generate.assert_not_called()


@pytest.mark.asyncio
async def test_valueerror_propagates_from_response_format(bedrock_config):
    """Test that ValueError from response_format propagates to the caller (Requirement 8.3)."""
    client = LLMClient(bedrock_config)
    request = TextRequest(prompt="Give me structured output", response_format=DummySchema)

    # The Bedrock client raises ValueError for response_format - it should propagate
    with patch.object(client._client, 'generate_text_streamed', new_callable=AsyncMock) as mock_streamed:
        mock_streamed.side_effect = ValueError(
            "generate_text_streamed does not support structured output (response_format). "
            "Use generate_text with the two-pass thinking approach as an alternative "
            "for structured output with large prompts."
        )

        with pytest.raises(ValueError) as exc_info:
            await client.generate_text_streamed(request)

        # Verify the error message suggests generate_text alternative
        assert "generate_text" in str(exc_info.value)
        assert "response_format" in str(exc_info.value) or "structured output" in str(exc_info.value)


@pytest.mark.asyncio
async def test_valueerror_not_caught_by_unified_client(bedrock_config):
    """Test that LLMClient does NOT catch or wrap the ValueError - it propagates directly."""
    client = LLMClient(bedrock_config)
    request = TextRequest(prompt="test", response_format=DummySchema)

    original_error = ValueError("test error from bedrock")

    with patch.object(client._client, 'generate_text_streamed', new_callable=AsyncMock) as mock_streamed:
        mock_streamed.side_effect = original_error

        with pytest.raises(ValueError) as exc_info:
            await client.generate_text_streamed(request)

        # The exact same exception object should propagate
        assert exc_info.value is original_error


@pytest.mark.asyncio
async def test_bedrock_delegation_passes_request_unchanged(bedrock_config):
    """Test that the request object is passed to the Bedrock client without modification."""
    client = LLMClient(bedrock_config)

    # Create a request with various parameters
    request = TextRequest(
        prompt="Translate this text",
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0,
        max_tokens=4096,
        top_p=0.9,
        system_prompt="You are a translator",
        use_cache=True,
    )

    with patch.object(client._client, 'generate_text_streamed', new_callable=AsyncMock) as mock_streamed:
        mock_streamed.return_value = TextResponse(
            text="result", model=BEDROCK_MODEL, stop_reason="end_turn",
            input_tokens=10, output_tokens=5,
        )
        await client.generate_text_streamed(request)

        # Verify the exact request object was passed (not a copy or modified version)
        args, kwargs = mock_streamed.call_args
        assert args[0] is request
