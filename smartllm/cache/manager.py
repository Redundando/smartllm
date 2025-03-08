from typing import Dict, Any, Optional
from cacherator import JSONCache
from logorator import Logger


class CacheManager(JSONCache):
    def __init__(self, data_id: str, directory: str = "data/llm"):
        super().__init__(data_id=data_id, directory=directory)

    def store_result(self, result: Dict[str, Any], key: str = "result"):
        setattr(self, key, result)
        self.json_cache_save()

    def get_result(self, key: str = "result") -> Optional[Dict[str, Any]]:
        return getattr(self, key, None)

    def has_result(self, key: str = "result") -> bool:
        return hasattr(self, key)