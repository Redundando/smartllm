"""JSON file-based cache for LLM responses"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from logorator import Logger


class JSONFileCache:
    """Simple JSON file cache for LLM responses
    
    Stores responses as JSON files in a cache directory with SHA256-based keys.
    
    Args:
        cache_dir: Directory to store cache files (default: .llm_cache)
    """
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _generate_key(self, **kwargs) -> str:
        """Generate cache key from request parameters
        
        Args:
            **kwargs: Request parameters to hash
            
        Returns:
            16-character hex string cache key
        """
        # Sort keys for consistent hashing
        sorted_items = sorted(kwargs.items())
        key_string = json.dumps(sorted_items, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response by key
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data dictionary or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception as e:
                Logger.note(f"Cache read failed for {cache_key}: {e}")
                return None
        return None
    
    def set(self, cache_key: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Store response in cache
        
        Args:
            cache_key: Cache key
            data: Response data to cache
            metadata: Optional metadata (prompt, model, etc.)
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_data = {
            "data": data,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))
    
    def clear(self, cache_key: Optional[str] = None):
        """Clear cache files
        
        Args:
            cache_key: If provided, only clear this specific cache entry.
                      If None, clear all cache files.
        """
        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
