from typing import Union, Optional, Dict, List, Any
from cacherator import Cached, JSONCache
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from logorator import Logger
import json

from .llm_provider import LLMProvider
from .perplexity_provider import PerplexityProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider


class LLMRequestState(Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class SmartLLM(JSONCache):
    TOKEN_PER_CHAR = 0.3
    MAX_INPUT_TOKENS = 10_000
    MAX_OUTPUT_TOKENS = 10_000

    _thread_pool = ThreadPoolExecutor(max_workers=10)

    PROVIDERS: Dict[str, LLMProvider] = {
            "perplexity": PerplexityProvider(),
            "anthropic" : AnthropicProvider(),
            "openai"    : OpenAIProvider()
    }

    def __init__(
            self,
            base: str = "",
            model: str = "",
            api_key: str = "",
            prompt: Union[str, List[str]] = "",
            max_input_tokens: Optional[int] = None,
            max_output_tokens: Optional[int] = None,
            output_type: str = "text",
            temperature: float = 0.2,
            top_p: float = 0.9,
            frequency_penalty: float = 1.0,
            presence_penalty: float = 0.0,
            system_prompt: Optional[str] = None,
            search_recency_filter: Optional[str] = None,
            return_citations: bool = False,
            json_mode: bool = False,
            json_schema: Optional[Dict[str, Any]] = None,
    ):
        # Set basic parameters first to construct identifier
        self.base = base
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else self.MAX_INPUT_TOKENS
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else self.MAX_OUTPUT_TOKENS
        self.output_type = output_type
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.search_recency_filter = search_recency_filter
        self.return_citations = return_citations
        self.json_mode = json_mode
        self.json_schema = json_schema

        # Call JSONCache.__init__ first to restore any cached values
        JSONCache.__init__(self, data_id=self.identifier, directory="data/llm")

        # Initialize state variables that shouldn't be cached
        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self._future = None
        self._client = None
        self._generation_result = None

    @property
    def identifier(self) -> str:
        prompt_str = str(self.prompt)
        truncated_prompt = prompt_str[:30] + "..." if len(prompt_str) > 30 else prompt_str
        base_id = f"{self.base}_{self.model}_{truncated_prompt}"

        # Create a more stable hash input
        hash_input = f"{self.base}_{self.model}_{str(self.prompt)}_{self.max_input_tokens}_{self.max_output_tokens}"
        hash_input += f"_{self.temperature}_{self.top_p}_{self.frequency_penalty}_{self.presence_penalty}"
        hash_input += f"_{self.system_prompt}_{self.search_recency_filter}"
        hash_input += f"_{self.return_citations}_{self.json_mode}"

        # For JSON schema, use a content hash instead of string representation
        if self.json_schema:
            schema_str = json.dumps(self.json_schema, sort_keys=True)
            schema_hash = sha256(schema_str.encode()).hexdigest()[:10]
            hash_input += f"_schema_{schema_hash}"

        _hash = sha256(hash_input.encode()).hexdigest()[:10]
        return f"{base_id}_{_hash}"

    @property
    def client(self) -> Any:
        if not self._client and self.base in self.PROVIDERS:
            provider = self.PROVIDERS[self.base]
            self._client = provider.create_client(
                    api_key=self.api_key
            )
        return self._client

    @Logger()
    def generate(self) -> 'SmartLLM':
        Logger.note(f"Starting LLM request for {self.base}/{self.model}")

        if self.state == LLMRequestState.PENDING:
            Logger.note("Request already in progress, not starting a new one")
            return self

        if self.state == LLMRequestState.COMPLETED:
            Logger.note("Request already completed, not starting a new one")
            return self

        self.state = LLMRequestState.PENDING
        self._future = self._thread_pool.submit(self._generate_in_background)
        return self

    @Cached()
    def _get_cached_generation(self) -> Dict[str, Any]:
        """
        Get cached generation result if available.
        If no cache exists, perform the API call and cache the result.
        """
        messages = self._prepare_messages()
        params = self._prepare_parameters(messages)

        provider = self.PROVIDERS[self.base]
        raw_response = provider.generate(
                client=self.client,
                model=self.model,
                messages=messages,
                params=params
        )

        # Create a serializable response using the provider's implementation
        return provider.create_serializable_response(raw_response, self.json_mode)

    def _prepare_messages(self) -> List[Dict[str, str]]:
        provider = self.PROVIDERS[self.base]
        return provider.prepare_messages(self.prompt, self.system_prompt)

    def _prepare_parameters(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        provider = self.PROVIDERS[self.base]

        # Add system_prompt to parameters for Anthropic
        if self.base == "anthropic":
            return provider.prepare_parameters(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    search_recency_filter=self.search_recency_filter,
                    json_mode=self.json_mode,
                    json_schema=self.json_schema,
                    system_prompt=self.system_prompt
            )
        else:
            return provider.prepare_parameters(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    search_recency_filter=self.search_recency_filter,
                    json_mode=self.json_mode,
                    json_schema=self.json_schema
            )

    def _generate_in_background(self) -> Dict[str, Any]:
        try:
            if self.base not in self.PROVIDERS:
                raise ValueError(f"Provider {self.base} not supported")

            Logger.note(f"Executing LLM request in background thread for {self.base}/{self.model}")

            # This will either get from cache or generate new
            self._generation_result = self._get_cached_generation()
            self.state = LLMRequestState.COMPLETED
            Logger.note(f"LLM request completed successfully")
            return self._generation_result

        except Exception as e:
            self.state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"LLM request failed: {str(e)}")
            raise

    @Logger()
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        if self.state == LLMRequestState.NOT_STARTED:
            Logger.note("Request not started yet, starting now")
            self.generate()

        if self.state == LLMRequestState.COMPLETED:
            return True

        if self.state == LLMRequestState.FAILED:
            return False

        if self._future is None:
            Logger.note("No future exists, request may not have been started properly")
            return False

        try:
            Logger.note(f"Waiting for LLM request to complete (timeout: {timeout})")
            self._future.result(timeout=timeout)
            return self.state == LLMRequestState.COMPLETED
        except Exception as e:
            Logger.note(f"Error while waiting for completion: {str(e)}")
            return False

    def is_completed(self) -> bool:
        return self.state == LLMRequestState.COMPLETED

    def is_pending(self) -> bool:
        return self.state == LLMRequestState.PENDING

    def is_failed(self) -> bool:
        return self.state == LLMRequestState.FAILED

    def get_error(self) -> Optional[str]:
        return self.error if self.is_failed() else None

    @property
    def content(self) -> Optional[str]:
        if not self.is_completed():
            return None
        return self._generation_result["content"]

    @property
    def json_content(self) -> Optional[Dict[str, Any]]:
        """Get the JSON content from the LLM response if available."""
        if not self.is_completed() or not self.json_mode:
            return None

        # Directly return the extracted JSON content that was cached
        return self._generation_result.get("json_content")

    @property
    def citations(self) -> List[str]:
        """Legacy property for backward compatibility. Use sources instead."""
        return self.sources

    @property
    def sources(self) -> List[str]:
        """Get citation sources from the LLM response if available."""
        if not self.is_completed():
            return []
        return self._generation_result.get("citations", [])

    @property
    def usage(self) -> Optional[Dict[str, int]]:
        if not self.is_completed():
            return None
        return self._generation_result.get("usage", {})

    @Logger()
    @Cached()
    def count_tokens(self) -> int:
        """
        Count tokens for the current prompt using the provider's token counting API
        if available, otherwise fall back to the character-based estimation.
        """
        if self.base not in self.PROVIDERS:
            raise ValueError(f"Provider {self.base} not supported")

        provider = self.PROVIDERS[self.base]

        try:
            messages = self._prepare_messages()
            return provider.count_tokens(
                    client=self.client,
                    model=self.model,
                    messages=messages,
                    system_prompt=self.system_prompt
            )
        except NotImplementedError:
            # Fall back to character-based estimation
            if isinstance(self.prompt, str):
                return self.estimate_tokens(self.prompt)
            else:
                return sum(self.estimate_tokens(msg) for msg in self.prompt)

    @Cached()
    def estimate_tokens(self, text: str) -> int:
        return int(len(text) * self.TOKEN_PER_CHAR)

    @Logger()
    @Cached()
    def list_available_models(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List available models for the current provider.
        """
        if self.base not in self.PROVIDERS:
            raise ValueError(f"Provider {self.base} not supported")

        provider = self.PROVIDERS[self.base]

        try:
            return provider.list_models(client=self.client, limit=limit)
        except NotImplementedError:
            Logger.note(f"Provider {self.base} does not support listing models")
            return []

    @staticmethod
    def convert_schema(schema: Any, provider: str = None) -> Dict[str, Any]:
        """
        Convert a schema (possibly a Pydantic model) to a provider-specific JSON schema.

        Args:
            schema: A Pydantic model, dict schema, or other schema object
            provider: Optional provider name for provider-specific adjustments

        Returns:
            A JSON schema compatible with the specified provider
        """
        # Handle Pydantic models
        if hasattr(schema, "model_json_schema"):
            # Pydantic model
            base_schema = schema.model_json_schema()
        elif isinstance(schema, dict):
            # Already a dict schema
            base_schema = schema
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        # Adjust schema based on provider requirements
        if provider == "anthropic":
            # Anthropic doesn't need adjustments currently
            return base_schema
        elif provider == "openai":
            # OpenAI doesn't need adjustments currently
            return base_schema
        elif provider == "perplexity":
            # Perplexity expects a wrapper with "schema"
            return {"schema": base_schema}
        else:
            # No provider specified or unknown provider
            return base_schema