"""
SmartLLM - A unified async Python wrapper for multiple LLM providers

Supports AWS Bedrock, OpenAI, and provides a unified interface for seamless switching.

Recommended usage:
    from smartllm import LLMClient, TextRequest
    
    async with LLMClient(provider="openai") as client:
        response = await client.generate_text(TextRequest(prompt="Hello"))
"""

# Shared models
from .models import (
    TextRequest,
    TextResponse,
    MessageRequest,
    Message,
    StreamChunk,
)

# Unified client (primary interface)
from .unified import LLMClient, LLMConfig

# Bedrock-specific exception classes (re-exported so consumers can catch
# `BedrockError` without reaching into the provider package). Importing
# from the bedrock package is gated on the boto3 dependency being present;
# we wrap in a try/except so OpenAI-only installs still work.
try:
    from .bedrock.exceptions import (
        BedrockError,
        BedrockStreamError,
        BedrockStreamTimeoutError,
    )
except ImportError:  # pragma: no cover - optional dependency
    BedrockError = None  # type: ignore[assignment]
    BedrockStreamError = None  # type: ignore[assignment]
    BedrockStreamTimeoutError = None  # type: ignore[assignment]

# Defaults (users can modify these)
from . import defaults

# Provider-specific clients available but not in main exports
# Advanced users can import: from smartllm.bedrock import BedrockLLMClient
# Advanced users can import: from smartllm.openai import OpenAILLMClient

__version__ = "0.1.24"
__author__ = "Arved Klöhn"

__all__ = [
    # Core models (shared)
    "TextRequest",
    "TextResponse", 
    "MessageRequest",
    "Message",
    "StreamChunk",
    
    # Unified client (primary interface)
    "LLMClient",
    "LLMConfig",

    # Bedrock-specific exceptions (None when bedrock extras not installed)
    "BedrockError",
    "BedrockStreamError",
    "BedrockStreamTimeoutError",

    # Defaults module
    "defaults",
]