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

# Defaults (users can modify these)
from . import defaults

# Provider-specific clients available but not in main exports
# Advanced users can import: from smartllm.bedrock import BedrockLLMClient
# Advanced users can import: from smartllm.openai import OpenAILLMClient

__version__ = "0.1.0"
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
    
    # Defaults module
    "defaults",
]