"""Two-level cache: local JSON files (fast) + DynamoDB (shared, TTL)"""

from typing import Optional, Dict, Any
from logorator import Logger
from .cache import JSONFileCache

DEFAULT_TTL_DAYS = 365


class TwoLevelCache:
    """Cache that reads local first, falls back to DynamoDB, writes to both.

    Args:
        cache_dir: Local cache directory (default: .llm_cache)
        dynamo_table_name: DynamoDB table name. If None, only local cache is used.
        ttl_days: TTL for DynamoDB entries in days (default: 30)
    """

    def __init__(self, cache_dir: str = ".llm_cache", dynamo_table_name: Optional[str] = None, ttl_days: float = DEFAULT_TTL_DAYS):
        self.local = JSONFileCache(cache_dir=cache_dir)
        self.ttl_days = ttl_days
        self._dynamo = None

        if dynamo_table_name:
            try:
                from dynamorator import DynamoDBStore
                self._dynamo = DynamoDBStore(table_name=dynamo_table_name, silent=True)
                Logger.note(f"DynamoDB cache enabled: {dynamo_table_name}")
            except ImportError:
                Logger.note("dynamorator not installed; DynamoDB cache disabled. Install with: pip install dynamorator")

    def _generate_key(self, **kwargs) -> str:
        return self.local._generate_key(**kwargs)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        # 1. Local
        result = self.local.get(cache_key)
        if result is not None:
            return result

        # 2. DynamoDB
        if self._dynamo:
            result = self._dynamo.get(cache_key)
            if result is not None:
                Logger.note(f"DynamoDB cache hit [{cache_key[:8]}], writing to local")
                self.local.set(cache_key, result.get("data", {}), result.get("metadata"))
                return result

        return None

    def set(self, cache_key: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.local.set(cache_key, data, metadata)
        if self._dynamo:
            self._dynamo.put(cache_key, {"data": data, "metadata": metadata or {}}, ttl_days=self.ttl_days)

    def clear(self, cache_key: Optional[str] = None):
        self.local.clear(cache_key)
        if self._dynamo:
            if cache_key:
                self._dynamo.delete(cache_key)
            else:
                Logger.note("TwoLevelCache.clear(all) only clears local; DynamoDB entries expire via TTL")
