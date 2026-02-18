"""OpenAI provider

Provides OpenAILLMClient for OpenAI API access with support for:
- GPT-4, GPT-3.5, and other OpenAI models
- Automatic retry with exponential backoff
- Response caching
- Optional concurrent request limiting
- Streaming responses
- Structured output with Pydantic models
"""

from .openai_client import OpenAILLMClient
from .config import OpenAIConfig

__all__ = ["OpenAILLMClient", "OpenAIConfig"]