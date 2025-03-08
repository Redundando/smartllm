from typing import Dict, List, Any, Callable, Optional
from queue import Queue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from logorator import Logger

from .execution.state import LLMRequestState


class BackgroundStreamProcessor:
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self.result = None
        self.chunk_queue = Queue()
        self.completed_event = threading.Event()
        self.future = None
        self.callback_history = []

    def process(
            self,
            stream_func: Callable,
            callback: Optional[Callable[[str], None]] = None,
            *args, **kwargs
    ):
        if self.state == LLMRequestState.PENDING:
            return

        self.state = LLMRequestState.PENDING

        def internal_callback(chunk: str):
            self.chunk_queue.put(chunk)
            self.callback_history.append(chunk)
            if callback:
                try:
                    callback(chunk)
                except Exception as e:
                    Logger.note(f"Error in user callback: {str(e)}")

        def process_stream():
            try:
                result = stream_func(*args, callback=internal_callback, **kwargs)
                self.result = result
                self.state = LLMRequestState.COMPLETED
                self.completed_event.set()
                return result
            except Exception as e:
                self.error = str(e)
                self.state = LLMRequestState.FAILED
                self.completed_event.set()
                raise

        self.future = self.executor.submit(process_stream)
        return self

    def is_completed(self) -> bool:
        return self.state == LLMRequestState.COMPLETED

    def is_failed(self) -> bool:
        return self.state == LLMRequestState.FAILED

    def is_pending(self) -> bool:
        return self.state == LLMRequestState.PENDING

    def get_error(self) -> Optional[str]:
        return self.error if self.is_failed() else None

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        try:
            return self.completed_event.wait(timeout)
        except Exception as e:
            Logger.note(f"Error waiting for completion: {str(e)}")
            return False

    def get_result(self) -> Any:
        return self.result

    def get_chunks(self, block: bool = False, timeout: Optional[float] = None) -> List[str]:
        chunks = []
        while not self.chunk_queue.empty():
            try:
                chunks.append(self.chunk_queue.get(block=block, timeout=timeout))
                self.chunk_queue.task_done()
            except Empty:
                break
        return chunks

    def replay_chunks(self, callback: Callable[[str], None]):
        for chunk in self.callback_history:
            try:
                callback(chunk)
            except Exception as e:
                Logger.note(f"Error in callback during replay: {str(e)}")