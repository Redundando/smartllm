from typing import Dict, Optional, Any
from logorator import Logger

from .providers import (
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    PerplexityProvider
)


class ProviderManager:
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {
            "anthropic": AnthropicProvider(),
            "openai": OpenAIProvider(),
            "perplexity": PerplexityProvider()
        }
        self.clients: Dict[str, Any] = {}

    def get_provider(self, base: str) -> LLMProvider:
        if base not in self.providers:
            raise ValueError(f"Provider {base} not supported")
        return self.providers[base]

    def get_client(self, base: str, api_key: str, base_url: Optional[str] = None) -> Any:
        client_key = f"{base}_{api_key}_{base_url}"

        if client_key in self.clients:
            return self.clients[client_key]

        provider = self.get_provider(base)
        client = provider.create_client(api_key=api_key, base_url=base_url)

        self.clients[client_key] = client
        return client