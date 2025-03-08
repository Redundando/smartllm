from typing import Union, Optional, Dict, List, Any, Callable
from cacherator import Cached
from concurrent.futures import Future
from logorator import Logger

from .config import Configuration
from .provider_manager import ProviderManager
from .cache.manager import CacheManager
from .execution.executor import RequestExecutor
from .execution.state import LLMRequestState
from .streaming.background import BackgroundStreamer


class SmartLLM:
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
            stream=stream,
        )

        self.provider_manager = ProviderManager()
        self.executor = RequestExecutor()
        self.cache = CacheManager(data_id=self.config.identifier)

        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self._future = None
        self._generation_result = None

        if self.config.stream:
            self._streamer = BackgroundStreamer(self.executor.thread_pool)
            if self.cache.has_result("stream_result"):
                self._streamer._cached_result = self.cache.get_result("stream_result")
                self._streamer._generation_result = self._streamer._cached_result
                self._streamer.state = LLMRequestState.COMPLETED
                self.state = LLMRequestState.COMPLETED
        else:
            self._streamer = None

            if self.cache.has_result():
                self._generation_result = self.cache.get_result()
                self.state = LLMRequestState.COMPLETED

    def __str__(self):
        return self.config.identifier

    @property
    def client(self) -> Any:
        return self.provider_manager.get_client(
            base=self.config.base,
            api_key=self.config.api_key
        )

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

        if self.config.stream:
            try:
                self._handle_streaming_request()
            except Exception as e:
                self.state = LLMRequestState.FAILED
                self.error = str(e)
                Logger.note(f"LLM request failed: {str(e)}")
                raise
        else:
            self._future = self.executor.submit(self._generate_in_background)

        return self

    def _handle_streaming_request(self) -> None:
        if not self._streamer:
            raise ValueError("Streamer not initialized")

        if self._streamer.is_completed():
            self._generation_result = self._streamer._generation_result
            self.state = LLMRequestState.COMPLETED
            return

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
            system_prompt=self.config.system_prompt if self.config.base == "anthropic" else None,
            stream=True
        )

        self._streamer.generate(
            base=self.config.base,
            provider=provider,
            client=self.client,
            model=self.config.model,
            messages=messages,
            params=params
        )

        self.state = self._streamer.state
        self.error = self._streamer.error

        if self.is_completed():
            self.cache.store_result(self._streamer._generation_result, "stream_result")

    @Logger()
    def generate_streaming(self, callback: Optional[Callable[[str], None]] = None) -> 'SmartLLM':
        Logger.note(f"Starting streaming LLM request for {self.config.base}/{self.config.model}")

        if not self._streamer:
            raise ValueError("Streamer not initialized")

        if self._streamer.is_completed():
            Logger.note("Using cached streaming result")
            self._streamer.replay_chunks(callback if callback else self._streamer.handle_chunk)
            self._generation_result = self._streamer._generation_result
            self.state = LLMRequestState.COMPLETED
            return self

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
            system_prompt=self.config.system_prompt if self.config.base == "anthropic" else None,
            stream=True
        )

        self.state = LLMRequestState.PENDING

        try:
            self._streamer.generate(
                base=self.config.base,
                provider=provider,
                client=self.client,
                model=self.config.model,
                messages=messages,
                params=params,
                callback=callback
            )

            self.state = self._streamer.state
            self.error = self._streamer.error
            self._generation_result = self._streamer._generation_result

            if self.is_completed():
                self.cache.store_result(self._generation_result, "stream_result")

        except Exception as e:
            self.state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"Streaming LLM request failed: {str(e)}")
            raise

        return self

    @Cached()
    def _get_cached_generation(self) -> Dict[str, Any]:
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

            result = self._get_cached_generation()
            self._generation_result = result
            self.state = LLMRequestState.COMPLETED
            self.cache.store_result(result)

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

        if self.config.stream:
            return self._streamer.wait_for_completion(timeout)

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
        if self.config.stream and self._streamer:
            return self._streamer.is_failed()
        return self.state == LLMRequestState.FAILED

    def is_completed(self) -> bool:
        if self.config.stream and self._streamer:
            return self._streamer.is_completed()
        return self.state == LLMRequestState.COMPLETED

    def is_pending(self) -> bool:
        if self.config.stream and self._streamer:
            return self._streamer.is_pending()
        return self.state == LLMRequestState.PENDING

    def get_error(self) -> Optional[str]:
        if self.config.stream and self._streamer:
            return self._streamer.get_error()
        return self.error

    @property
    def content(self) -> str:
        if self.config.stream and self._streamer and self._streamer.is_completed():
            return self._streamer.get_content()
        if not self._generation_result:
            return ""
        return self._generation_result.get("content", "")