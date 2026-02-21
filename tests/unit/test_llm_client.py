"""Unit tests for LLMClient (unified client)"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from smartllm import LLMClient, LLMConfig, TextRequest, MessageRequest, Message

UNIT_MODEL = "gpt-4o-mini"

@pytest.fixture
def openai_config():
    return LLMConfig(provider="openai", api_key="test-key", default_model=UNIT_MODEL, temperature=0, max_tokens=100)


@pytest.mark.asyncio
async def test_client_initialization(openai_config):
    """Test client initializes correctly"""
    client = LLMClient(openai_config)

    assert client.config == openai_config
    assert client.provider == "openai"
    assert client._client is not None


@pytest.mark.asyncio
async def test_cache_hit_skips_api_call(openai_config, mock_openai_response):
    """Test that cache hit prevents API call"""
    client = LLMClient(openai_config)

    cache_key = client._client.cache._generate_key(
        api_type="responses",
        model=UNIT_MODEL,
        prompt="test",
        max_tokens=100,
        instructions=None,
        reasoning_effort=None,
        response_format=None
    )
    client._client.cache.set(cache_key, {
        "text": "cached response",
        "model": UNIT_MODEL,
        "stop_reason": "stop",
        "input_tokens": 5,
        "output_tokens": 3,
        "metadata": {},
        "structured_data": None,
    })

    with patch.object(client._client, '_invoke_with_retry', new_callable=AsyncMock) as mock_invoke:
        with patch.object(client._client, '_init_client', new_callable=AsyncMock):
            from smartllm.openai.responses_api import ResponsesAPI
            from smartllm.openai.config import OpenAIConfig
            client._client.responses_api = ResponsesAPI(MagicMock(), client._client.config, client._client.cache)
            client._client.client = MagicMock()  # mark as initialized
            response = await client.generate_text(TextRequest(prompt="test", temperature=0))

        mock_invoke.assert_not_called()
        assert response.text == "cached response"


@pytest.mark.asyncio
async def test_clear_cache_flag(openai_config):
    """Test clear_cache flag removes cached entry"""
    client = LLMClient(openai_config)

    cache_key = client._client.cache._generate_key(
        api_type="responses",
        model=UNIT_MODEL,
        prompt="test",
        max_tokens=100,
        instructions=None,
        reasoning_effort=None,
        response_format=None
    )
    client._client.cache.set(cache_key, {"text": "cached"})
    assert client._client.cache.get(cache_key) is not None

    with patch.object(client._client, '_invoke_with_retry', new_callable=AsyncMock):
        with patch.object(client._client, '_init_client', new_callable=AsyncMock):
            from smartllm.openai.responses_api import ResponsesAPI
            client._client.responses_api = ResponsesAPI(MagicMock(), client._client.config, client._client.cache)
            client._client.client = MagicMock()
            try:
                await client.generate_text(TextRequest(prompt="test", temperature=0, clear_cache=True))
            except:
                pass

    assert client._client.cache.get(cache_key) is None


@pytest.mark.asyncio
async def test_message_request_delegates_to_provider(openai_config, mock_openai_response):
    """Test MessageRequest delegates to provider client"""
    client = LLMClient(openai_config)

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?"),
    ]

    with patch.object(client._client, 'send_message', new_callable=AsyncMock) as mock_send:
        mock_send.return_value = MagicMock(text="response")
        await client.send_message(MessageRequest(messages=messages, temperature=1.0, use_cache=False))
        mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_providers():
    """Test getting list of available providers"""
    providers = LLMClient.get_available_providers()

    assert "openai" in providers
    assert "bedrock" in providers
    assert len(providers) >= 2


@pytest.mark.asyncio
async def test_list_models_for_provider():
    """Test listing models for a specific provider"""
    with patch('smartllm.openai.OpenAILLMClient.list_available_models', new_callable=AsyncMock) as mock_list:
        with patch('smartllm.openai.OpenAILLMClient._init_client', new_callable=AsyncMock):
            mock_list.return_value = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

            models = await LLMClient.list_models_for_provider("openai", api_key="test-key")

            assert len(models) > 0
            assert any("gpt" in m for m in models)
