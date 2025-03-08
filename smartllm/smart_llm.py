from typing import Union, Optional, Dict, List, Any, Callable
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
from .llm_streamer import SmartLLMStreamer, LLMRequestState


class SmartLLM(JSONCache):
    TOKEN_PER_CHAR = 0.3
    MAX_INPUT_TOKENS = 10_000
    MAX_OUTPUT_TOKENS = 10_000

    _thread_pool = ThreadPoolExecutor(max_workers=10)

    PROVIDERS: Dict[str, LLMProvider] = {
        "perplexity": PerplexityProvider(),
        "anthropic": AnthropicProvider(),
        "openai": OpenAIProvider()
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
            stream: bool = False,
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
        self.stream = stream

        # Initialize streaming handler if streaming is enabled
        self._streamer = SmartLLMStreamer() if stream else None

        # Call JSONCache.__init__ first to restore any cached values
        # Skip caching for streaming requests
        if not stream:
            JSONCache.__init__(self, data_id=self.identifier, directory="data/llm")

        # Initialize state variables that shouldn't be cached
        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self._future = None
        self._client = None
        self._generation_result = None

    def __str__(self):
        return self.identifier

    @property
    def identifier(self) -> str:
        prompt_str = str(self.prompt)
        truncated_prompt = prompt_str[:30] + "..." if len(prompt_str) > 30 else prompt_str
        base_id = f"{self.base}_{self.model}_{truncated_prompt}"

        # Create a more stable hash input
        hash_input = f"{self.base}_{self.model}_{str(self.prompt)}_{self.max_input_tokens}_{self.max_output_tokens}"
        hash_input += f"_{self.temperature}_{self.top_p}_{self.frequency_penalty}_{self.presence_penalty}"
        hash_input += f"_{self.system_prompt}_{self.search_recency_filter}"
        hash_input += f"_{self.return_citations}_{self.json_mode}_{self.stream}"

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

        # Handle streaming requests differently
        if self.stream:
            try:
                self._handle_streaming_request()
            except Exception as e:
                self.state = LLMRequestState.FAILED
                self.error = str(e)
                Logger.note(f"LLM request failed: {str(e)}")
                raise
        else:
            # Use thread pool for non-streaming requests
            self._future = self._thread_pool.submit(self._generate_in_background)

        return self

    def _handle_streaming_request(self) -> None:
        """
        Internal method to handle streaming requests.
        Uses the SmartLLMStreamer to process streaming.
        """
        if self.base not in self.PROVIDERS:
            raise ValueError(f"Provider {self.base} not supported for streaming")

        provider = self.PROVIDERS[self.base]
        messages = self._prepare_messages()
        params = self._prepare_parameters(messages)

        # Use the streamer to handle the streaming request
        self._generation_result = self._streamer.generate(
            base=self.base,
            provider=provider,
            client=self.client,
            model=self.model,
            messages=messages,
            params=params
        )

        # Update state based on streamer state
        self.state = self._streamer.state
        self.error = self._streamer.error

    @Logger()
    def generate_streaming(self, callback: Optional[Callable[[str], None]] = None) -> 'SmartLLM':
        """
        Generate a streaming response from the LLM, calling the callback with each chunk.

        Args:
            callback: Function that will be called with each text chunk. 
                     If None, uses a default callback that logs the chunks.

        Returns:
            self for chaining
        """
        Logger.note(f"Starting streaming LLM request for {self.base}/{self.model}")

        if self.base not in self.PROVIDERS:
            raise ValueError(f"Provider {self.base} not supported")

        if self.state == LLMRequestState.PENDING:
            Logger.note("Request already in progress, not starting a new one")
            return self

        if self.state == LLMRequestState.COMPLETED:
            Logger.note("Request already completed, not starting a new one")
            return self

        self.state = LLMRequestState.PENDING

        try:
            provider = self.PROVIDERS[self.base]
            messages = self._prepare_messages()
            params = self._prepare_parameters(messages)

            # Use the streamer to handle the streaming request
            self._generation_result = self._streamer.generate(
                base=self.base,
                provider=provider,
                client=self.client,
                model=self.model,
                messages=messages,
                params=params,
                callback=callback
            )

            # Update state based on streamer state
            self.state = self._streamer.state
            self.error = self._streamer.error

        except Exception as e:
            self.state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"Streaming LLM request failed: {str(e)}")
            raise

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
                system_prompt=self.system_prompt,
                stream=self.stream
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
                json_schema=self.json_schema,
                stream=self.stream
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

        # For streaming requests, delegate to the streamer
        if self.stream:
            return self._streamer.is_completed()

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
        """
        Check if the request has completed successfully.
        For streaming requests, delegates to the streamer.
        """
        if self.stream and self._streamer:
            return self._streamer.is_complete