from typing import Dict, List, Any, Callable, Optional
from queue import Queue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from logorator import Logger

from .streamer import Streamer
from ..execution.state import LLMRequestState


class BackgroundStreamer(Streamer):
    def __init__(self, thread_pool: ThreadPoolExecutor):
        super().__init__()
        self.thread_pool = thread_pool
        self.chunk_queue = Queue()
        self._future = None
        self.completed_event = threading.Event()

    def generate(
            self,
            base: str,
            provider: Any,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        if self.state == LLMRequestState.PENDING:
            return self._generation_result

        if self.state == LLMRequestState.COMPLETED:
            if callback and self._generation_result:
                content = self._generation_result.get("content", "")
                if content:
                    callback(content)
            return self._generation_result

        self.state = LLMRequestState.PENDING

        def internal_callback(chunk: str):
            self.chunk_queue.put(chunk)
            self._callback_history.append(chunk)
            # Debug log to confirm callback is being used
            Logger.note(f"BackgroundStreamer received chunk of {len(chunk)} chars")
            if callback:
                try:
                    # Directly pass the chunk to the user's callback
                    callback(chunk)
                except Exception as e:
                    Logger.note(f"Error in user callback: {str(e)}")

        self._future = self.thread_pool.submit(
            self._stream_in_background,
            base, provider, client, model, messages, params, internal_callback
        )

        return self._generation_result

    def _stream_in_background(
            self,
            base: str,
            provider: Any,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callback: Callable[[str], None]
    ) -> Dict[str, Any]:
        try:
            full_text = None

            if base == "anthropic":
                from ..streaming.provider_streamers.anthropic_streamer import AnthropicStreamer
                streamer = AnthropicStreamer()
                # Make sure we're passing the callback correctly
                full_text = streamer.stream(
                    client=client,
                    model=model,
                    messages=messages,
                    params=params,
                    callback=callback
                )
                self._generation_result = streamer.format_response(full_text)
            elif base == "openai":
                raise NotImplementedError("OpenAI streaming not yet implemented")
            elif base == "perplexity":
                raise NotImplementedError("Perplexity streaming not yet implemented")
            else:
                raise ValueError(f"Streaming not supported for provider: {base}")

            self.state = LLMRequestState.COMPLETED
            self.completed_event.set()
            Logger.note(f"Streaming LLM request completed successfully")

            return self._generation_result

        except Exception as e:
            self.state = LLMRequestState.FAILED
            self.error = str(e)
            self.completed_event.set()
            Logger.note(f"Streaming LLM request failed: {str(e)}")
            raise

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        if self._future is None:
            return False

        try:
            return self.completed_event.wait(timeout)
        except Exception as e:
            Logger.note(f"Error waiting for streaming completion: {str(e)}")
            return False

    def get_chunks(self, block: bool = False, timeout: Optional[float] = None) -> List[str]:
        chunks = []
        while not self.chunk_queue.empty():
            try:
                chunks.append(self.chunk_queue.get(block=block, timeout=timeout))
                self.chunk_queue.task_done()
            except Empty:
                break
        return chunks