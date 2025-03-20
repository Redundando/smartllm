from typing import Dict, Optional, Any

from .async_providers import (
    AsyncLLMProvider,
    AsyncAnthropicProvider
)

class AsyncProviderManager:
    def __init__(self):
        self.providers: Dict[str, AsyncLLMProvider] = {
            "anthropic": AsyncAnthropicProvider()
        }
        self.clients: Dict[str, Any] = {}

    def get_provider(self, base: str) -> AsyncLLMProvider:
        if base not in self.providers:
            raise ValueError(f"Provider {base} not supported")
        return self.providers[base]

    async def get_client(self, base: str, api_key: str, base_url: Optional[str] = None) -> Any:
        client_key = f"{base}_{api_key}_{base_url}"

        if client_key in self.clients:
            return self.clients[client_key]

        provider = self.get_provider(base)
        client = await provider.create_client(api_key=api_key, base_url=base_url)

        self.clients[client_key] = client
        return client