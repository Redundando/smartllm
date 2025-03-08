from typing import Dict, List, Any, Callable, Optional
from logorator import Logger

from ..execution.state import LLMRequestState


class Streamer:
    def __init__(self):
        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self._generation_result = None
        self._cached_result = None
        self._callback_history = []

    def set_cache_info(self, cache_obj, identifier: str) -> bool:
        if hasattr(cache_obj, "_cached_stream_result"):
            self._cached_result = cache_obj._cached_stream_result
            self._generation_result = self._cached_result
            self.state = LLMRequestState.COMPLETED
            return True
        return False

    def save_to_cache(self, cache_obj):
        if self.is_completed() and self._generation_result:
            cache_obj._cached_stream_result = self._generation_result
            cache_obj.json_cache_save()

    def handle_chunk(self, chunk: str) -> None:
        Logger.note(f"Received chunk: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")

    def is_completed(self) -> bool:
        return self.state == LLMRequestState.COMPLETED

    def is_pending(self) -> bool:
        return self.state == LLMRequestState.PENDING

    def is_failed(self) -> bool:
        return self.state == LLMRequestState.FAILED

    def get_error(self) -> Optional[str]:
        return self.error if self.is_failed() else None

    def get_content(self) -> str:
        if self._generation_result:
            return self._generation_result.get("content", "")
        return ""

    def replay_chunks(self, callback: Callable[[str], None]):
        if self.is_completed() and self._callback_history:
            for chunk in self._callback_history:
                callback(chunk)
        elif self.is_completed() and self._generation_result:
            callback(self._generation_result.get("content", ""))