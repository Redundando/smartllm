from typing import Any, Callable, Dict, List, Optional, Union

from cacherator import Cached, JSONCache
from logorator import Logger

from .config import Configuration
from .execution.state import LLMRequestState
from .provider_manager import ProviderManager


def default_streaming_callback(chunk: str, accumulated: str) -> None:
    Logger.note(f"{chunk}")


class SmartLLM(JSONCache):
    DEFAULT_TTL = 7

    def __init__(self, base: str = "", model: str = "", api_key: str = "", prompt: Union[str, List[str]] = "", stream: bool = False, **kwargs):
        # Pass all kwargs to Configuration - it will use what it needs and ignore the rest
        self.config = Configuration(base=base, model=model, api_key=api_key, prompt=prompt, **kwargs)

        # Extract JSONCache parameters
        ttl = kwargs.get("ttl", self.DEFAULT_TTL)
        clear_cache = kwargs.get("clear_cache", False)

        super().__init__(data_id=self.config.identifier, directory="data/llm", ttl=ttl, clear_cache=clear_cache)

        self.cached_config = self.config.safe_config
        self.provider_manager = ProviderManager()
        self._state = LLMRequestState.COMPLETED if hasattr(self, "result") and self.result else None
        self.error = None

        # Streaming-related fields
        self.stream_enabled = stream
        self.streaming_callbacks = [] if stream else None

    def __str__(self):
        return self.config.identifier

    @property
    def client(self) -> Any:
        return self.provider_manager.get_client(base=self.config.base, api_key=self.config.api_key)

    def _prepare_request(self):
        provider = self.provider_manager.get_provider(self.config.base)
        messages = provider.prepare_messages(self.config.prompt, self.config.system_prompt)

        params = provider.prepare_parameters(model=self.config.model, messages=messages, max_tokens=self.config.max_output_tokens, temperature=self.config.temperature, top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty, presence_penalty=self.config.presence_penalty, search_recency_filter=self.config.search_recency_filter, json_mode=self.config.json_mode,
                json_schema=self.config.json_schema, system_prompt=self.config.system_prompt)

        return provider, params

    @Logger()
    def execute(self, callback: Optional[Callable[[str, str], None]] = None) -> 'SmartLLM':
        Logger.note(f"Starting LLM request for {self.config.base}/{self.config.model}")

        if self._state == LLMRequestState.COMPLETED:
            Logger.note("Using cached result")
            return self

        # Add callback if streaming is enabled and callback is provided
        if self.stream_enabled and callback:
            self.streaming_callbacks.append(callback)
        elif self.stream_enabled and not self.streaming_callbacks:
            # Add default callback if none exists
            self.streaming_callbacks.append(default_streaming_callback)

        try:
            if self.stream_enabled:
                self._execute_streaming_request()
            else:
                self._execute_request()

            self._state = LLMRequestState.COMPLETED
            self.json_cache_save()

        except Exception as e:
            self._state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"LLM request failed: {str(e)}")

        return self

    @Cached()
    def _get_llm_response(self) -> Dict[str, Any]:
        provider, params = self._prepare_request()

        raw_response = provider.generate(client=self.client, model=self.config.model, messages=params.get("messages", []), params=params)

        return provider.create_response(raw_response, self.config.json_mode)

    @Cached()
    def _get_streaming_llm_response(self) -> Dict[str, Any]:
        provider = self.provider_manager.get_provider(self.config.base)
        messages = provider.prepare_messages(self.config.prompt, self.config.system_prompt)

        params = provider.prepare_parameters(model=self.config.model, messages=messages, max_tokens=self.config.max_output_tokens, temperature=self.config.temperature, top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty, presence_penalty=self.config.presence_penalty, search_recency_filter=self.config.search_recency_filter, json_mode=self.config.json_mode,
                json_schema=self.config.json_schema, system_prompt=self.config.system_prompt, stream=True)

        return provider.generate_stream(client=self.client, model=self.config.model, messages=messages, params=params, callbacks=self.streaming_callbacks)

    def _execute_request(self) -> Dict[str, Any]:
        Logger.note(f"Executing LLM request for {self.config.base}/{self.config.model}")
        result = self._get_llm_response()
        self.result = result
        Logger.note("LLM request completed successfully")
        return result

    def _execute_streaming_request(self) -> Dict[str, Any]:
        Logger.note(f"Executing streaming request for {self.config.base}/{self.config.model}")
        result = self._get_streaming_llm_response()
        self.result = result
        Logger.note("Streaming request completed successfully")
        return result

    def generate(self) -> 'SmartLLM':
        return self.execute()

    def is_failed(self) -> bool:
        return self._state == LLMRequestState.FAILED

    def is_completed(self) -> bool:
        return self._state == LLMRequestState.COMPLETED

    def get_error(self) -> Optional[str]:
        return self.error

    def _get_result_property(self, property_name: str, default=None):
        if not hasattr(self, "result") or not self.result:
            return default
        return self.result.get(property_name, default)

    @property
    def _content(self) -> str:
        return self._get_result_property("content", "")

    @property
    def _json_content(self) -> Optional[Dict[str, Any]]:
        return self._get_result_property("json_content")

    @property
    def response(self) -> Union[str, Dict[str, Any]]:
        if self.config.json_mode and self._json_content:
            return self._json_content
        return self._content

    @property
    def sources(self) -> List[str]:
        return self._get_result_property("citations", [])

    @property
    def usage(self) -> Dict[str, int]:
        default_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return self._get_result_property("usage", default_usage)
