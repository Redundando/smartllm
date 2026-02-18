"""Configuration module for OpenAI LLM Wrapper"""

import os
from typing import Optional
from ..defaults import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_MAX_RETRY_DELAY,
    OPENAI_DEFAULT_MODEL,
    OPENAI_DEFAULT_TOP_P,
)


class OpenAIConfig:
    """Configuration for OpenAI
    
    Configuration priority: Constructor args > Environment variables > Defaults
    
    Args:
        api_key: OpenAI API key
        organization: OpenAI organization ID (optional)
        default_model: Default OpenAI model (default: gpt-4o-mini)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens
        top_p: Nucleus sampling parameter
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        max_retry_delay: Maximum retry delay in seconds
        max_concurrent: Maximum concurrent requests (optional)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        default_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        max_retry_delay: Optional[float] = None,
        max_concurrent: Optional[int] = None,
    ):
        # OpenAI Credentials
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        
        # Default model configurations
        self.default_model = default_model or os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("OPENAI_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        self.top_p = top_p if top_p is not None else float(os.getenv("OPENAI_TOP_P", str(OPENAI_DEFAULT_TOP_P)))
        
        # Retry configurations
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("OPENAI_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
        self.retry_delay = retry_delay if retry_delay is not None else float(os.getenv("OPENAI_RETRY_DELAY", str(DEFAULT_RETRY_DELAY)))
        self.max_retry_delay = max_retry_delay if max_retry_delay is not None else float(os.getenv("OPENAI_MAX_RETRY_DELAY", str(DEFAULT_MAX_RETRY_DELAY)))
        
        # Rate limit configurations
        self.max_concurrent = max_concurrent if max_concurrent is not None else (int(os.getenv("OPENAI_MAX_CONCURRENT")) if os.getenv("OPENAI_MAX_CONCURRENT") else None)

    def validate(self) -> bool:
        """Validate that required OpenAI API key is present
        
        Returns:
            True if API key is valid
            
        Raises:
            ValueError: If API key is missing
        """
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it via:\n"
                "1. Environment variable: OPENAI_API_KEY\n"
                "2. Constructor argument: OpenAIConfig(api_key='...')"
            )
        return True
