"""Shared data models for SmartLLM"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type, Callable
from pydantic import BaseModel


@dataclass
class TextRequest:
    """Request for text generation
    
    Attributes:
        prompt: The input text prompt
        model: Model ID to use (optional, uses default if None)
        temperature: Sampling temperature 0-1 (optional)
        max_tokens: Maximum output tokens (optional)
        top_p: Nucleus sampling parameter (optional)
        top_k: Top-k sampling parameter (optional, Bedrock only)
        system_prompt: System prompt to set context (optional)
        stream: Enable streaming response (default: False)
        response_format: Pydantic model for structured output (optional)
        use_cache: Enable response caching (default: True)
        clear_cache: Clear cache before request (default: False)
        api_type: OpenAI API type - "responses" (default) or "chat_completions"
    """
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    system_prompt: Optional[str] = None
    stream: bool = False
    response_format: Optional[Type[BaseModel]] = None
    use_cache: bool = True
    clear_cache: bool = False
    api_type: str = "responses"
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" - reasoning models only
    on_progress: Optional[Callable] = None

    def __str__(self):
        return self.prompt[:150] + "..." if len(self.prompt) > 150 else self.prompt


@dataclass
class Message:
    """A message in a conversation
    
    Attributes:
        role: Message role ("user" or "assistant")
        content: Message content text
    """
    role: str  # "user" or "assistant"
    content: str


@dataclass
class MessageRequest:
    """Request for multi-turn conversation
    
    Attributes:
        messages: List of conversation messages
        model: Model ID to use (optional, uses default if None)
        temperature: Sampling temperature 0-1 (optional)
        max_tokens: Maximum output tokens (optional)
        system_prompt: System prompt to set context (optional)
        stream: Enable streaming response (default: False)
        response_format: Pydantic model for structured output (optional)
        use_cache: Enable response caching (default: True)
        clear_cache: Clear cache before request (default: False)
        api_type: OpenAI API type - "responses" (default) or "chat_completions"
    """
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    stream: bool = False
    response_format: Optional[Type[BaseModel]] = None
    use_cache: bool = True
    clear_cache: bool = False
    api_type: str = "responses"
    on_progress: Optional[Callable] = None

    def __str__(self):
        last = self.messages[-1].content if self.messages else ""
        return f"{len(self.messages)} messages | {last[:50]}"


@dataclass
class TextResponse:
    """Response from LLM
    
    Attributes:
        text: Generated text content
        model: Model ID that generated the response
        stop_reason: Reason generation stopped
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        metadata: Additional response metadata
        structured_data: Parsed structured output (if response_format was used)
    """
    text: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_data: Optional[BaseModel] = None
    cache_source: str = "miss"


@dataclass
class StreamChunk:
    """A chunk from a streaming response
    
    Attributes:
        text: Partial text content
        model: Model ID generating the stream
        metadata: Additional chunk metadata
    """
    text: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
