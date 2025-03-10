from typing import Union, Optional, Dict, List, Any, Tuple
from cacherator import JSONCache, Cached
from logorator import Logger

from .config import Configuration
from .provider_manager import ProviderManager
from .execution.executor import RequestExecutor
from .execution.state import LLMRequestState


class SmartLLM(JSONCache):
    # Default values as class constants
    DEFAULT_TTL = 7
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_P = 0.9
    DEFAULT_FREQUENCY_PENALTY = 1.0
    DEFAULT_PRESENCE_PENALTY = 0.0
    DEFAULT_OUTPUT_TYPE = "text"

    def __init__(
            self,
            base: str = "",
            model: str = "",
            api_key: str = "",
            prompt: Union[str, List[str]] = "",
            **kwargs
    ):
        # Create configuration using core parameters + kwargs for flexibility
        self.config = Configuration(
                base=base,
                model=model,
                api_key=api_key,
                prompt=prompt,
                **kwargs
        )

        # Extract cache-specific parameters
        ttl = kwargs.get("ttl", self.DEFAULT_TTL)
        clear_cache = kwargs.get("clear_cache", False)

        # Initialize base JSONCache
        super().__init__(
                data_id=self.config.identifier,
                directory="data/llm",
                ttl=ttl,
                clear_cache=clear_cache
        )

        self.cached_config = self.config.safe_config
        self.provider_manager = ProviderManager()
        self.executor = RequestExecutor()
        self._state = LLMRequestState.NOT_STARTED
        self.error = None
        self._future = None

        # If we already have results in cache, mark as completed
        if hasattr(self, "result") and self.result:
            self._state = LLMRequestState.COMPLETED

    def __str__(self):
        return self.config.identifier

    @property
    def client(self) -> Any:
        return self.provider_manager.get_client(
                base=self.config.base,
                api_key=self.config.api_key
        )

    def _prepare_request(self) -> Tuple[Any, Dict[str, Any]]:
        provider = self.provider_manager.get_provider(self.config.base)
        messages = provider.prepare_messages(self.config.prompt, self.config.system_prompt)

        # Only pass system_prompt in parameters for Anthropic
        use_system_in_params = self.config.base == "anthropic"

        params = provider.prepare_parameters(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                search_recency_filter=self.config.search_recency_filter,
                json_mode=self.config.json_mode,
                json_schema=self.config.json_schema,
                system_prompt=self.config.system_prompt if use_system_in_params else None
        )

        return provider, params

    @Logger()
    def generate(self) -> 'SmartLLM':
        Logger.note(f"Starting LLM request for {self.config.base}/{self.config.model}")

        if self._state == LLMRequestState.PENDING:
            Logger.note("Request already in progress")
            return self

        if self._state == LLMRequestState.COMPLETED:
            Logger.note("Request already completed")
            return self

        self._state = LLMRequestState.PENDING
        self._future = self.executor.submit(self._execute_request)

        return self

    @Cached()
    def _get_llm_response(self) -> Dict[str, Any]:
        provider, params = self._prepare_request()

        raw_response = provider.generate(
                client=self.client,
                model=self.config.model,
                messages=params.get("messages", []),
                params=params
        )

        return provider.create_serializable_response(raw_response, self.config.json_mode)

    def _execute_request(self) -> Dict[str, Any]:
        try:
            Logger.note(f"Executing LLM request for {self.config.base}/{self.config.model}")

            result = self._get_llm_response()

            self.result = result
            self._state = LLMRequestState.COMPLETED

            self.json_cache_save()

            Logger.note("LLM request completed successfully")
            return result

        except Exception as e:
            self._state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"LLM request failed: {str(e)}")
            raise

    @Logger()
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        if self._state == LLMRequestState.NOT_STARTED:
            Logger.note("Request not started yet, starting now")
            self.generate()

        if self._state == LLMRequestState.COMPLETED:
            return True

        if self._state == LLMRequestState.FAILED:
            return False

        if self._future is None:
            Logger.note("No future exists, request may not have been started properly")
            return False

        try:
            Logger.note(f"Waiting for LLM request to complete (timeout: {timeout})")
            self._future.result(timeout=timeout)
            return self._state == LLMRequestState.COMPLETED
        except Exception as e:
            Logger.note(f"Error while waiting for completion: {str(e)}")
            return False

    def is_failed(self) -> bool:
        return self._state == LLMRequestState.FAILED

    def is_completed(self) -> bool:
        return self._state == LLMRequestState.COMPLETED

    def is_pending(self) -> bool:
        return self._state == LLMRequestState.PENDING

    def get_error(self) -> Optional[str]:
        return self.error

    def _get_result_property(self, property_name: str, default=None):
        if not hasattr(self, "result") or not self.result:
            return default
        return self.result.get(property_name, default)

    @property
    def content(self) -> str:
        return self._get_result_property("content", "")

    @property
    def json_content(self) -> Optional[Dict[str, Any]]:
        return self._get_result_property("json_content")

    @property
    def sources(self) -> List[str]:
        return self._get_result_property("citations", [])

    @property
    def usage(self) -> Dict[str, int]:
        default_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return self._get_result_property("usage", default_usage)