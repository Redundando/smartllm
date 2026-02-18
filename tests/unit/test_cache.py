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
