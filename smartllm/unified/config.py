"""Unified configuration for all LLM providers"""

import os
from typing import Optional, Literal


class LLMConfig:
    """Unified configuration for LLM providers
    
    Supports both OpenAI and AWS Bedrock providers with automatic provider detection.
    Configuration priority: Constructor args > Environment variables > Defaults
    
    Args:
        provider: Provider name ("openai" or "bedrock"). Auto-detected if None.
        api_key: API key for OpenAI
        default_model: Default model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens
        top_p: Nucleus sampling parameter
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        max_retry_delay: Maximum retry delay in seconds
        max_concurrent: Maximum concurrent requests
        organization: OpenAI organization ID (OpenAI only)
        aws_access_key_id: AWS access key (Bedrock only)
        aws_secret_access_key: AWS secret key (Bedrock only)
        aws_session_token: AWS session token (Bedrock only)
        aws_region: AWS region (Bedrock only)
        top_k: Top-k sampling parameter (Bedrock only)
    """
    
    def __init__(
        self,
        provider: Optional[Literal["openai", "bedrock"]] = None,
        # Common settings
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        max_retry_delay: Optional[float] = None,
        max_concurrent: Optional[int] = None,
        # OpenAI specific
        organization: Optional[str] = None,
        # Bedrock specific
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        top_k: Optional[int] = None,
        # Cache specific
        dynamo_table_name: Optional[str] = None,
        cache_ttl_days: Optional[float] = None,
    ):
        # Auto-detect provider if not specified
        if provider is None:
            provider = self._detect_provider()
        
        self.provider = provider
        
        # Common settings
        self.api_key = api_key
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.max_concurrent = max_concurrent
        
        # OpenAI specific
        self.organization = organization
        
        # Bedrock specific
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region = aws_region
        self.top_k = top_k

        # Cache specific
        self.dynamo_table_name = dynamo_table_name
        self.cache_ttl_days = cache_ttl_days
    
    def _detect_provider(self) -> str:
        """Auto-detect provider from environment variables
        
        Returns:
            Provider name ("openai" or "bedrock")
        """
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_REGION"):
            return "bedrock"
        else:
            # Default to OpenAI
            return "openai"
    
    def to_openai_config(self):
        """Convert to OpenAI-specific config
        
        Returns:
            OpenAIConfig instance
        """
        from ..openai import OpenAIConfig
        return OpenAIConfig(
            api_key=self.api_key,
            organization=self.organization,
            default_model=self.default_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            max_retry_delay=self.max_retry_delay,
            max_concurrent=self.max_concurrent,
        )
    
    def to_bedrock_config(self):
        """Convert to Bedrock-specific config
        
        Returns:
            BedrockConfig instance
        """
        from ..bedrock import BedrockConfig
        return BedrockConfig(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            aws_region=self.aws_region,
            default_model=self.default_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            max_retry_delay=self.max_retry_delay,
            max_concurrent=self.max_concurrent,
        )
