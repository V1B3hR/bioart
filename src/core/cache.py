"""
Bounded cache with TTL for performance optimization.

Implements M1 requirement: "Introduce bounded caches (e.g., sequence scoring, transforms) with TTL"
"""

import time
import threading
from typing import Any, Dict, Optional, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass

from src.core import get_logger


logger = None
def get_logger_cached():
    global logger
    if logger is None:
        from src.core import get_logger
        logger = get_logger("cache")
    return logger


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expires_at: float


class BoundedTTLCache:
    """
    Thread-safe bounded cache with time-to-live (TTL) expiration.

    Features:
    - Maximum size limit (LRU eviction when full)
    - TTL-based expiration
    - Thread-safe operations
    - Hit/miss statistics
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live in seconds
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            now = time.time()

            # Check if expired
            if now >= entry.expires_at:
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Custom TTL (uses default if None)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        expires_at = time.time() + ttl

        with self._lock:
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = []

        with self._lock:
            for key, entry in self._cache.items():
                if now >= entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_requests": total_requests,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0


class CachedFunction:
    """
    Decorator for caching function results.

    Example:
        @cached(max_size=100, ttl_seconds=60)
        def expensive_function(x, y):
            return x + y
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300,
        key_func: Optional[Callable] = None
    ):
        """
        Initialize cached function decorator.

        Args:
            max_size: Maximum cache size
            ttl_seconds: TTL in seconds
            key_func: Optional function to generate cache key from args
        """
        self.cache = BoundedTTLCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self.key_func = key_func or self._default_key_func

    @staticmethod
    def _default_key_func(*args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)

    def __call__(self, func: Callable) -> Callable:
        """Wrap function with caching."""
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{self.key_func(*args, **kwargs)}"

            # Try to get from cache
            result = self.cache.get(cache_key)
            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            self.cache.set(cache_key, result)

            return result

        # Add cache management methods
        wrapper.cache = self.cache
        wrapper.cache_clear = self.cache.clear
        wrapper.cache_stats = self.cache.get_stats

        return wrapper


# Global caches for common use cases
_sequence_cache: Optional[BoundedTTLCache] = None
_transform_cache: Optional[BoundedTTLCache] = None


def get_sequence_cache() -> BoundedTTLCache:
    """
    Get global sequence scoring cache.

    Returns:
        BoundedTTLCache instance
    """
    global _sequence_cache
    if _sequence_cache is None:
        from src.core import get_config
        config = get_config()
        _sequence_cache = BoundedTTLCache(
            max_size=config.performance.cache_max_size,
            ttl_seconds=config.performance.cache_ttl_seconds
        )
    return _sequence_cache


def get_transform_cache() -> BoundedTTLCache:
    """
    Get global transform cache.

    Returns:
        BoundedTTLCache instance
    """
    global _transform_cache
    if _transform_cache is None:
        from src.core import get_config
        config = get_config()
        _transform_cache = BoundedTTLCache(
            max_size=config.performance.cache_max_size,
            ttl_seconds=config.performance.cache_ttl_seconds
        )
    return _transform_cache


def cached(
    max_size: int = 1000,
    ttl_seconds: float = 300,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.

    Args:
        max_size: Maximum cache size
        ttl_seconds: TTL in seconds
        key_func: Optional function to generate cache key

    Example:
        @cached(max_size=100, ttl_seconds=60)
        def score_sequence(sequence: str) -> float:
            # expensive computation
            return score
    """
    return CachedFunction(max_size=max_size, ttl_seconds=ttl_seconds, key_func=key_func)
