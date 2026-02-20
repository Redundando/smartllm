"""Pytest configuration and shared fixtures"""

import pytest
import os
import tomllib
from pathlib import Path
from smartllm import LLMConfig

_models_config = tomllib.loads((Path(__file__).parent / "models.toml").read_text())
DEFAULT_MODEL = _models_config["default"]
_models = {m["id"]: m for m in _models_config["models"]}


def pytest_addoption(parser):
    parser.addoption("--model", default=None, help="Model to use for all tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "reasoning: mark test as requiring a reasoning-capable model")
    config.addinivalue_line("markers", "openai: mark test as OpenAI-only")
    config.addinivalue_line("markers", "no_responses_api: mark test as unsupported with Responses API")
    if config.option.__dict__.get("model") is None:
        models = list(_models.keys())
        print("\nAvailable models:")
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")
        print(f"Select model [1-{len(models)}] (default: {DEFAULT_MODEL}): ", end="", flush=True)
        with open("CON", "r") as tty:
            choice = tty.readline().strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            config.option.model = models[int(choice) - 1]
        else:
            config.option.model = DEFAULT_MODEL


def pytest_collection_modifyitems(config, items):
    model = config.getoption("--model")
    model_cfg = _models.get(model, {})
    if not model_cfg.get("reasoning", False):
        skip = pytest.mark.skip(reason=f"{model} does not support reasoning")
        for item in items:
            if item.get_closest_marker("reasoning"):
                item.add_marker(skip)
    if model_cfg.get("responses_api", False):
        skip = pytest.mark.skip(reason=f"{model} uses Responses API which does not support this feature")
        for item in items:
            if item.get_closest_marker("no_responses_api"):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def test_model(request):
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def test_provider(test_model):
    return _models.get(test_model, {}).get("provider", "openai")


@pytest.fixture(scope="session")
def test_api_type(test_model):
    return "responses" if _models.get(test_model, {}).get("responses_api", False) else "chat_completions"


@pytest.fixture
def llm_config(test_model, test_provider):
    """LLM config with provider for testing"""
    return LLMConfig(
        provider=test_provider,
        api_key=os.getenv("OPENAI_API_KEY", "test-key") if test_provider == "openai" else None,
        default_model=test_model,
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
