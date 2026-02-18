"""Unit tests for LLMConfig (unified configuration)"""

import pytest
from smartllm import LLMConfig


def test_config_auto_detects_openai(monkeypatch):
    """Test config auto-detects OpenAI from environment"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    
    config = LLMConfig()
    
    assert config.provider == "openai"


def test_config_explicit_provider():
    """Test config with explicit provider"""
    config = LLMConfig(
        provider="openai",
        api_key="test-key",
        default_model="gpt-4o",
        temperature=0.5,
        max_tokens=1000,
    )
    
    assert config.provider == "openai"
    assert config.api_key == "test-key"
    assert config.default_model == "gpt-4o"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000


def test_config_defaults_to_openai(monkeypatch):
    """Test config defaults to OpenAI when no env vars set"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    
    config = LLMConfig()
    
    assert config.provider == "openai"


def test_config_to_openai_config():
    """Test conversion to OpenAI-specific config"""
    config = LLMConfig(
        provider="openai",
        api_key="test-key",
        default_model="gpt-4o",
        temperature=0.7,
    )
    
    openai_config = config.to_openai_config()
    
    assert openai_config.api_key == "test-key"
    assert openai_config.default_model == "gpt-4o"
    assert openai_config.temperature == 0.7
