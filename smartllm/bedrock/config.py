"""Configuration module for AWS Bedrock LLM Wrapper"""

import os
from typing import Optional
import boto3
from botocore.exceptions import NoCredentialsError
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
        aws_region: AWS region. Resolution chain (boto3-compatible):
            constructor arg > AWS_REGION > AWS_DEFAULT_REGION > package default.
        default_model: Default Bedrock model ID (inference profile ID for
            modern Anthropic models, e.g. `us.anthropic.claude-sonnet-4-6`)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        max_retry_delay: Maximum retry delay in seconds
        max_concurrent: Maximum concurrent requests (optional)
        read_timeout: HTTP read timeout in seconds (default: 300)
        connect_timeout: HTTP connect timeout in seconds (default: 10)
        stream_total_timeout: Maximum seconds a streaming request may run from
            stream open until completion. Triggers `BedrockStreamTimeoutError`
            with kind="total". Set to 0 or negative to disable. Default: 900.
        stream_first_chunk_timeout: Maximum seconds to wait for the first
            event after a streaming request is accepted by Bedrock. Triggers
            `BedrockStreamTimeoutError` with kind="first_chunk". Set to 0 or
            negative to disable. Default: 60.
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
        read_timeout: Optional[int] = None,
        connect_timeout: Optional[int] = None,
        stream_total_timeout: Optional[float] = None,
        stream_first_chunk_timeout: Optional[float] = None,
    ):
        # AWS Credentials: explicit args > environment variables
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        # Region resolution mirrors boto3:
        #   explicit arg > AWS_REGION > AWS_DEFAULT_REGION > package default.
        # Many AWS environments (Lambda, ECS, EC2 with default profile)
        # only set AWS_DEFAULT_REGION, so honoring both is important.
        self.aws_region = (
            aws_region
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or BEDROCK_DEFAULT_REGION
        )
        
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
        
        # HTTP timeout configurations
        self.read_timeout = read_timeout if read_timeout is not None else int(os.getenv("BEDROCK_READ_TIMEOUT", "300"))
        self.connect_timeout = connect_timeout if connect_timeout is not None else int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "10"))

        # Streaming-specific timeouts. These guard against the failure modes
        # where Bedrock's streaming API accepts a request but then either
        # delays the first event indefinitely (TPM-saturation queueing) or
        # half-closes the connection mid-stream. Both are documented at:
        # tmp/smartllm-observability-bug-report.md (issues 2 and 3).
        self.stream_total_timeout = (
            stream_total_timeout
            if stream_total_timeout is not None
            else float(os.getenv("BEDROCK_STREAM_TOTAL_TIMEOUT", "900"))
        )
        self.stream_first_chunk_timeout = (
            stream_first_chunk_timeout
            if stream_first_chunk_timeout is not None
            else float(os.getenv("BEDROCK_STREAM_FIRST_CHUNK_TIMEOUT", "60"))
        )

    def validate(self) -> bool:
        """Validate that required AWS credentials are present
        
        Returns:
            True if credentials are valid
            
        Raises:
            ValueError: If required credentials are missing
        """
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            try:
                boto3.Session(region_name=self.aws_region).get_credentials().get_frozen_credentials()
            except (NoCredentialsError, AttributeError):
                raise ValueError(
                    "AWS credentials not found. Please provide them via:\n"
                    "1. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                    "2. Constructor arguments: BedrockConfig(aws_access_key_id='...', aws_secret_access_key='...')\n"
                    "3. An IAM role attached to your EC2/ECS/Lambda environment"
                )
        return True

    def get_credentials(self) -> dict:
        """Get AWS credentials as a dictionary
        
        Returns:
            Dictionary with AWS credentials (region_name, aws_access_key_id, 
            aws_secret_access_key, and optionally aws_session_token)
        """
        creds = {"region_name": self.aws_region}
        if self.aws_access_key_id:
            creds["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            creds["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            creds["aws_session_token"] = self.aws_session_token
        return creds
