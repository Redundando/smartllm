from typing import Union, Optional, Dict, List, Any,  Tuple
from cacherator import JSONCache, Cached
from logorator import Logger

from .config import Configuration
from .provider_manager import ProviderManager
from .execution.executor import RequestExecutor
from .execution.state import LLMRequestState


class SmartLLM(JSONCache):
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
            ttl: int = 7,
            clear_cache: bool = False
    ):
        self.config = Configuration(
            base=base,
            model=model,
            api_key=api_key,
            prompt=prompt,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            output_type=output_type,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            system_prompt=system_prompt,
            search_recency_filter=search_recency_filter,
            return_citations=return_citations,
            json_mode=json_mode,
            json_schema=json_schema,
        )

        super().__init__(
            data_id=self.config.identifier,
            directory="data/llm",
            ttl=ttl,
            clear_cache=clear_cache
        )

        self.cached_config = self.config.safe_config

        self.provider_manager = ProviderManager()
        self.executor = RequestExecutor()
        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self._future = None

        if hasattr(self, "result") and self.result:
            self.state = LLMRequestState.COMPLETED

    def __str__(self):
        return self.config.identifier

    def generate_response(self) -> 'SmartLLM':
        return self.generate()

    @property
    def client(self) -> Any:
        return self.provider_manager.get_client(
            base=self.config.base,
            api_key=self.config.api_key
        )

    def _prepare_request_params(self) -> Tuple[Any, List[Dict[str, str]], Dict[str, Any]]:
        provider = self.provider_manager.get_provider(self.config.base)
        messages = provider.prepare_messages(self.config.prompt, self.config.system_prompt)
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
            system_prompt=self.config.system_prompt if self.config.base == "anthropic" else None
        )
        return provider, messages, params

    @Logger()
    def generate(self) -> 'SmartLLM':
        Logger.note(f"Starting LLM request for {self.config.base}/{self.config.model}")

        if self.state == LLMRequestState.PENDING:
            Logger.note("Request already in progress")
            return self

        if self.state == LLMRequestState.COMPLETED:
            Logger.note("Request already completed")
            return self

        self.state = LLMRequestState.PENDING
        self._future = self.executor.submit(self._generate_in_background)

        return self

    @Cached()
    def _get_llm_response(self) -> Dict[str, Any]:
        provider, messages, params = self._prepare_request_params()

        raw_response = provider.generate(
            client=self.client,
            model=self.config.model,
            messages=messages,
            params=params
        )

        return provider.create_serializable_response(raw_response, self.config.json_mode)

    def _generate_in_background(self) -> Dict[str, Any]:
        try:
            Logger.note(f"Executing LLM request in background thread for {self.config.base}/{self.config.model}")

            result = self._get_llm_response()

            self.result = result
            self.state = LLMRequestState.COMPLETED

            self.json_cache_save()

            Logger.note(f"LLM request completed successfully")
            return result

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

    def is_failed(self) -> bool:
        return self.state == LLMRequestState.FAILED

    def is_completed(self) -> bool:
        return self.state == LLMRequestState.COMPLETED

    def is_pending(self) -> bool:
        return self.state == LLMRequestState.PENDING

    def get_error(self) -> Optional[str]:
        return self.error

    @property
    def content(self) -> str:
        if not hasattr(self, "result") or not self.result:
            return ""
        return self.result.get("content", "")

    @property
    def json_content(self) -> Optional[Dict[str, Any]]:
        if not hasattr(self, "result") or not self.result:
            return None
        return self.result.get("json_content")

    @property
    def sources(self) -> List[str]:
        if not hasattr(self, "result") or not self.result:
            return []
        return self.result.get("citations", [])

    @property
    def usage(self) -> Dict[str, int]:
        if not hasattr(self, "result") or not self.result:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return self.result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})