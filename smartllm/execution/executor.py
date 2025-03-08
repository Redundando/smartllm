from typing import Dict, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from logorator import Logger


class RequestExecutor:
    def __init__(self, max_workers: int = 10):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, func: Callable, *args, **kwargs):
        return self.thread_pool.submit(func, *args, **kwargs)