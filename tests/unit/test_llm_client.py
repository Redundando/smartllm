"""Unit tests for LLMClient (unified client)"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from smartllm import LLMClient, LLMConfig, TextRequest, MessageRequest, Message


@pytest.mark.asyncio
async def test_client_initialization(llm_config):
    """Test client initializes correctly"""
    client = LLMClient(llm_config)
    
    assert client.config == llm_config
    assert client.provider == "openai"
    assert client._client is not None  # Provider client is initialized


@pytest.mark.asyncio
async def test_cache_hit_skips_api_call(llm_config, mock_openai_response):
    """Test that cache hit prevents API call"""
    client = LLMClient(llm_config)
    
    # Pre-populate cache in the underlying provider client
    cache_key = client._client.cache._generate_key(
        api_type="responses",
        model="gpt-4o-mini",
        prompt="test",
        max_tokens=100,
        instructions=None,
        reasoning_effort=None,
        response_format=None
    )
    client._client.cache.set(cache_key, {
        "text": "cached response",
        "model": "gpt-4o-mini",
        "stop_reason": "stop",
        "input_tokens": 5,
        "output_tokens": 3,
        "metadata": {},
        "structured_data": None,
    })
    
    # Mock the provider client's API call
    with patch.object(client._client, '_invoke_with_retry', new_callable=AsyncMock) as mock_invoke:
        request = TextRequest(prompt="test", temperature=0)
        response = await client.generate_text(request)
        
        # API should not be called
        mock_invoke.assert_not_called()
        assert response.text == "cached response"


@pytest.mark.asyncio
async def test_clear_cache_flag(llm_config):
    """Test clear_cache flag removes cached entry"""
    client = LLMClient(llm_config)
    
    # Pre-populate cache
    cache_key = client._client.cache._generate_key(
        api_type="responses",
        model="gpt-4o-mini",
        prompt="test",
        max_tokens=100,
        instructions=None,
        reasoning_effort=None,
        response_format=None
    )
    client._client.cache.set(cache_key, {"text": "cached"})
    
    assert client._client.cache.get(cache_key) is not None
    
    # Request with clear_cache should remove it
    with patch.object(client._client, '_invoke_with_retry', new_callable=AsyncMock):
        request = TextRequest(prompt="test", temperature=0, clear_cache=True)
        try:
            await client.generate_text(request)
        except:
            pass  # We don't care if it fails, just that cache was cleared
    
    assert client._client.cache.get(cache_key) is None


@pytest.mark.asyncio
async def test_message_request_delegates_to_provider(llm_config, mock_openai_response):
    """Test MessageRequest delegates to provider client"""
    client = LLMClient(llm_config)
    
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?"),
    ]
    
    with patch.object(client._client, 'send_message', new_callable=AsyncMock) as mock_send:
        mock_send.return_value = MagicMock(text="response")
        
        request = MessageRequest(messages=messages, temperature=1.0, use_cache=False)
        await client.send_message(request)
        
        # Verify provider client was called
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
    # Mock to avoid needing real credentials
    with patch('smartllm.openai.OpenAILLMClient.list_available_models', new_callable=AsyncMock) as mock_list:
        mock_list.return_value = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        
        models = await LLMClient.list_models_for_provider("openai", api_key="test-key")
        
        assert len(models) > 0
        assert any("gpt" in m for m in models)
