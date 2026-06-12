"""Integration-test fixtures: model selection.

The model-picker lives here (not in the root conftest) so that unit-test
runs never block on stdin. The picker is also TTY-aware: it only prompts
when stdin is an interactive terminal. In non-TTY contexts (CI pipelines,
piped invocations, IDE-driven test runners) it falls back to the default
model defined in `tests/models.toml`.

Selection precedence:
    1. `--model=<id>` on the command line
    2. Interactive picker (if `sys.stdin.isatty()`)
    3. Default from `models.toml`
"""

import os
import sys
import tomllib
from pathlib import Path

import pytest

from smartllm import LLMConfig

_MODELS_TOML = Path(__file__).parent.parent / "models.toml"
_models_config = tomllib.loads(_MODELS_TOML.read_text())
DEFAULT_MODEL: str = _models_config["default"]
_models = {m["id"]: m for m in _models_config["models"]}


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default=None,
        help="Model ID to run integration tests against (see tests/models.toml)",
    )


def _pick_model_interactively() -> str:
    """Prompt the user for a model. Caller must have already verified TTY."""
    models = list(_models.keys())
    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    print(f"Select model [1-{len(models)}] (default: {DEFAULT_MODEL}): ", end="", flush=True)
    choice = sys.stdin.readline().strip()
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    return DEFAULT_MODEL


def pytest_configure(config):
    """Resolve which model to use as soon as integration tests are configured."""
    if config.option.model is not None:
        return  # user passed --model explicitly

    if sys.stdin.isatty():
        config.option.model = _pick_model_interactively()
    else:
        config.option.model = DEFAULT_MODEL
        # Make the implicit choice visible in CI/non-TTY logs
        print(f"\n[integration] No --model and stdin not a TTY; using default: {DEFAULT_MODEL}")


def pytest_collection_modifyitems(config, items):
    """Skip tests incompatible with the chosen model."""
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
