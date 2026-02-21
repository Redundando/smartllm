"""Unified LLM client that works with multiple providers"""

from typing import Optional, AsyncIterator, Union
from .config import LLMConfig
from ..bedrock import BedrockLLMClient
from ..openai import OpenAILLMClient
from ..models import TextRequest, MessageRequest, TextResponse, StreamChunk


class LLMClient:
    """Unified async client for multiple LLM providers"""
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        **kwargs
    ):
        """Initialize unified LLM client
        
        Args:
            config: LLMConfig instance. If None, creates default config.
            provider: Provider name ("openai" or "bedrock"). Overrides config.provider.
            max_concurrent: Max concurrent requests.
            **kwargs: Additional config parameters passed to LLMConfig.
        """
        # Create config if not provided
        if config is None:
            config = LLMConfig(provider=provider, **kwargs)
        elif provider is not None:
            config.provider = provider
        
        self.config = config
        self._max_concurrent = max_concurrent
        
        # Initialize the appropriate provider client
        if config.provider == "openai":
            provider_config = config.to_openai_config()
            self._client = OpenAILLMClient(provider_config, max_concurrent=max_concurrent, dynamo_table_name=config.dynamo_table_name, cache_ttl_days=config.cache_ttl_days)
        elif config.provider == "bedrock":
            provider_config = config.to_bedrock_config()
            self._client = BedrockLLMClient(provider_config, max_concurrent=max_concurrent, dynamo_table_name=config.dynamo_table_name, cache_ttl_days=config.cache_ttl_days)
        else:
            raise ValueError(f"Unknown provider: {config.provider}. Use 'openai' or 'bedrock'.")
    
    @property
    def provider(self) -> str:
        """Get current provider name"""
        return self.config.provider
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def close(self):
        """Close the client connections"""
        await self._client.close()
    
    async def generate_text(self, request: TextRequest) -> TextResponse:
        """Generate text from a prompt
        
        Args:
            request: TextRequest with prompt and parameters
            
        Returns:
            TextResponse with generated text
        """
        return await self._client.generate_text(request)
    
    async def generate_text_stream(self, request: TextRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation
        
        Args:
            request: TextRequest with prompt and parameters
            
        Yields:
            StreamChunk objects with partial text
        """
        async for chunk in self._client.generate_text_stream(request):
            yield chunk
    
    async def send_message(self, request: MessageRequest) -> TextResponse:
        """Send a message in a conversation
        
        Args:
            request: MessageRequest with message history
            
        Returns:
            TextResponse with assistant's response
        """
        return await self._client.send_message(request)
    
    async def send_message_stream(self, request: MessageRequest) -> AsyncIterator[StreamChunk]:
        """Stream a conversation message
        
        Args:
            request: MessageRequest with message history
            
        Yields:
            StreamChunk objects with partial responses
        """
        async for chunk in self._client.send_message_stream(request):
            yield chunk
    
    async def list_available_models(self) -> list:
        """List all available models for the current provider
        
        Returns:
            List of model IDs or model summaries
        """
        return await self._client.list_available_models()
    
    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of all available providers
        
        Returns:
            List of provider names
        """
        return ["openai", "bedrock"]
    
    @staticmethod
    async def list_models_for_provider(provider: str, **config_kwargs) -> list:
        """List all available models for a specific provider
        
        Args:
            provider: Provider name ("openai" or "bedrock")
            **config_kwargs: Configuration parameters for the provider
            
        Returns:
            List of model IDs or model summaries
        """
        config = LLMConfig(provider=provider, **config_kwargs)
        async with LLMClient(config) as client:
            return await client.list_available_models()
