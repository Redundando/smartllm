"""Configuration module for AWS Bedrock LLM Wrapper"""

import os
from typing import Optional
from ..defaults import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_MAX_RETRY_DELAY,
    BEDROCK_DEFAULT_MODEL,
    BEDROCK_DEFAULT_REGION,
    BEDROCK_DEFAULT_TOP_P,
    BEDROCK_DEFAULT_TOP_K,
)


class BedrockConfig:
    """Configuration for AWS Bedrock
    
    Configuration priority: Constructor args > Environment variables > Defaults
    
    Args:
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token (optional)
        aws_region: AWS region (default: us-east-1)
        default_model: Default Bedrock model ID
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        max_retry_delay: Maximum retry delay in seconds
        max_concurrent: Maximum concurrent requests (optional)
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        default_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        max_retry_delay: Optional[float] = None,
        max_concurrent: Optional[int] = None,
    ):
        # AWS Credentials: explicit args > environment variables
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self.aws_region = aws_region or os.getenv("AWS_REGION", BEDROCK_DEFAULT_REGION)
        
        # Default model configurations
        self.default_model = default_model or os.getenv("BEDROCK_MODEL", BEDROCK_DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else float(os.getenv("BEDROCK_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("BEDROCK_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        self.top_p = top_p if top_p is not None else float(os.getenv("BEDROCK_TOP_P", str(BEDROCK_DEFAULT_TOP_P)))
        self.top_k = top_k if top_k is not None else int(os.getenv("BEDROCK_TOP_K", str(BEDROCK_DEFAULT_TOP_K)))
        
        # Retry configurations
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("BEDROCK_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
        self.retry_delay = retry_delay if retry_delay is not None else float(os.getenv("BEDROCK_RETRY_DELAY", str(DEFAULT_RETRY_DELAY)))
        self.max_retry_delay = max_retry_delay if max_retry_delay is not None else float(os.getenv("BEDROCK_MAX_RETRY_DELAY", str(DEFAULT_MAX_RETRY_DELAY)))
        
        # Rate limit configurations
        self.max_concurrent = max_concurrent if max_concurrent is not None else (int(os.getenv("BEDROCK_MAX_CONCURRENT")) if os.getenv("BEDROCK_MAX_CONCURRENT") else None)

    def validate(self) -> bool:
        """Validate that required AWS credentials are present
        
        Returns:
            True if credentials are valid
            
        Raises:
            ValueError: If required credentials are missing
        """
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError(
                "AWS credentials not found. Please provide them via:\n"
                "1. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                "2. Constructor arguments: BedrockConfig(aws_access_key_id='...', aws_secret_access_key='...')"
            )
        return True

    def get_credentials(self) -> dict:
        """Get AWS credentials as a dictionary
        
        Returns:
            Dictionary with AWS credentials (region_name, aws_access_key_id, 
            aws_secret_access_key, and optionally aws_session_token)
        """
        creds = {
            "region_name": self.aws_region,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
        }
        if self.aws_session_token:
            creds["aws_session_token"] = self.aws_session_token
        return creds
