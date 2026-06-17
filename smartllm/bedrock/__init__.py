"""AWS Bedrock provider

Provides BedrockLLMClient for AWS Bedrock LLM access with support for:
- Claude, Llama, Mistral, and Titan models
- Automatic retry with exponential backoff
- Response caching
- Concurrent request limiting per model
- Streaming responses
- Structured output with Pydantic models
"""

# Re-export everything from the original aws_bedrock_wrapper
from .bedrock_client import BedrockLLMClient
from .config import BedrockConfig
from .exceptions import (
    BedrockError,
    BedrockStreamError,
    BedrockStreamTimeoutError,
)
from ..models import TextRequest, TextResponse, MessageRequest, Message, StreamChunk

__all__ = [
    "BedrockLLMClient",
    "BedrockConfig",
    "BedrockError",
    "BedrockStreamError",
    "BedrockStreamTimeoutError",
    "TextRequest",
    "TextResponse",
    "MessageRequest", 
    "Message",
    "StreamChunk",
]