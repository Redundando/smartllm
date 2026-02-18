"""Unit tests for cache functionality"""

import pytest
import tempfile
import shutil
from pathlib import Path
from smartllm.utils import JSONFileCache


@pytest.fixture
def temp_cache():
    """Temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    cache = JSONFileCache(cache_dir=temp_dir)
    yield cache
    shutil.rmtree(temp_dir)


def test_cache_key_generation(temp_cache):
    """Test cache key is consistent for same inputs"""
    key1 = temp_cache._generate_key(model="gpt-4", prompt="test")
    key2 = temp_cache._generate_key(model="gpt-4", prompt="test")
    key3 = temp_cache._generate_key(model="gpt-4", prompt="different")
    
    assert key1 == key2
    assert key1 != key3


def test_cache_set_and_get(temp_cache):
    """Test setting and getting cache entries"""
    key = "test_key"
    data = {"text": "response", "tokens": 10}
    
    temp_cache.set(key, data)
    cached = temp_cache.get(key)
    
    assert cached is not None
    assert cached["data"] == data
    assert "cached_at" in cached


def test_cache_miss(temp_cache):
    """Test cache miss returns None"""
    result = temp_cache.get("nonexistent_key")
    assert result is None


def test_cache_clear_specific(temp_cache):
    """Test clearing specific cache entry"""
    key1 = "key1"
    key2 = "key2"
    
    temp_cache.set(key1, {"data": "1"})
    temp_cache.set(key2, {"data": "2"})
    
    temp_cache.clear(key1)
    
    assert temp_cache.get(key1) is None
    assert temp_cache.get(key2) is not None


def test_cache_clear_all(temp_cache):
    """Test clearing all cache entries"""
    temp_cache.set("key1", {"data": "1"})
    temp_cache.set("key2", {"data": "2"})
    
    temp_cache.clear()
    
    assert temp_cache.get("key1") is None
    assert temp_cache.get("key2") is None


@pytest.mark.asyncio
async def test_cache_stores_prompt(temp_cache):
    """Test that ResponsesAPI stores the prompt in cache metadata after an API call"""
    from unittest.mock import AsyncMock, MagicMock
    from smartllm.openai.responses_api import ResponsesAPI
    from smartllm.openai.config import OpenAIConfig
    from smartllm.models import TextRequest

    # Mock OpenAI response
    mock_usage = MagicMock(input_tokens=10, output_tokens=5, output_tokens_details=None, input_tokens_details=None)
    mock_response = MagicMock(output_text="Paris", status="completed", usage=mock_usage)

    mock_client = MagicMock()
    config = OpenAIConfig(api_key="test-key", default_model="gpt-4o-mini")
    api = ResponsesAPI(mock_client, config, temp_cache)

    prompt = "What is the capital of France?"
    request = TextRequest(prompt=prompt, temperature=0)

    await api.generate_text(request, AsyncMock(return_value=mock_response))

    cache_key = temp_cache._generate_key(
        api_type="responses",
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=config.max_tokens,
        instructions=None,
        reasoning_effort=None,
        response_format=None,
    )
    cached = temp_cache.get(cache_key)

    assert cached is not None
    assert cached["metadata"]["prompt"] == prompt
