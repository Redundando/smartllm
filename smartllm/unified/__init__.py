"""Unified LLM client

Provides LLMClient and LLMConfig for unified access to multiple LLM providers.

The unified client automatically routes requests to the appropriate provider
(OpenAI or AWS Bedrock) based on configuration, providing a consistent interface
regardless of the underlying provider.
"""

from .client import LLMClient
from .config import LLMConfig

__all__ = ["LLMClient", "LLMConfig"]