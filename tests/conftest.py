"""Pytest configuration and shared fixtures"""

import pytest
import os
from smartllm import LLMConfig


@pytest.fixture
def llm_config():
    """LLM config with OpenAI provider for testing"""
    return LLMConfig(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        default_model="gpt-4o-mini",
        temperature=0,
        max_tokens=100,
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {
                'content': 'Test response',
                'tool_calls': None
            })()
            self.finish_reason = 'stop'
    
    class MockUsage:
        prompt_tokens = 10
        completion_tokens = 5
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.usage = MockUsage()
    
    return MockResponse()
