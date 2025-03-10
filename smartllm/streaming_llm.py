from typing import Optional, List, Dict, Any, Callable, Union
from cacherator import Cached
from logorator import Logger

from .smart_llm import SmartLLM
from .execution.state import LLMRequestState


def default_streaming_callback(chunk: str, accumulated: str) -> None:
    Logger.note(f"Received chunk ({len(chunk)} chars): {chunk[:20]}...")


class StreamingLLM(SmartLLM):
    def __init__(
            self,
            base: str = "",
            model: str = "",
            api_key: str = "",
            prompt: Union[str, List[str]] = "",
            **kwargs
    ):
        super().__init__(base=base, model=model, api_key=api_key, prompt=prompt, **kwargs)
        self.streaming_callbacks: List[Callable[[str, str], None]] = []

    def stream(self, callback: Optional[Callable[[str, str], None]] = None) -> 'StreamingLLM':
        if callback:
            self.streaming_callbacks.append(callback)
        else:
            self.streaming_callbacks.append(default_streaming_callback)

        self._state = LLMRequestState.PENDING
        self._future = self.executor.submit(self._execute_streaming_request)

        return self

    def _execute_streaming_request(self) -> Dict[str, Any]:
        try:
            Logger.note(f"Starting streaming request for {self.config.model}")

            result = self._get_streaming_llm_response()

            self.result = result
            self._state = LLMRequestState.COMPLETED

            self.json_cache_save()

            return result
        except Exception as e:
            self._state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"Streaming request failed: {str(e)}")
            raise

    @Cached()
    def _get_streaming_llm_response(self) -> Dict[str, Any]:
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
                system_prompt=self.config.system_prompt
        )

        return provider.generate_stream(
                client=self.client,
                model=self.config.model,
                messages=messages,
                params=params,
                callbacks=self.streaming_callbacks
        )