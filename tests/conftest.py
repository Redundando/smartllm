"""Pytest configuration and shared fixtures.

This root conftest only declares fixtures and markers that apply to *all*
tests (unit + integration). Anything tied to the per-run model picker —
the `--model` CLI option, the interactive picker, and the
`test_model`/`test_provider`/`test_api_type`/`llm_config` fixtures — lives
in `tests/integration/conftest.py` so unit-test runs never block on stdin.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "reasoning: mark test as requiring a reasoning-capable model")
    config.addinivalue_line("markers", "openai: mark test as OpenAI-only")
    config.addinivalue_line("markers", "no_responses_api: mark test as unsupported with Responses API")


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
