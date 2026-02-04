"""Redis-based caching for performance optimization."""

import asyncio
import json
import pickle
import time
from typing import Any, Dict, Optional, Union

import redis.asyncio as aioredis

from vllm_orchestrator.observability.logging import get_logger
from vllm_orchestrator.observability.metrics import metrics_registry

logger = get_logger(__name__)


class CacheManager:
    """Redis-based cache manager with automatic serialization."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,  # 1 hour
        key_prefix: str = "vllm_orchestrator:",
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.redis: Optional[aioredis.Redis] = None
        self._cache_stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0,
            "deletes": 0,
        }

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )

            # Test connection
            await self.redis.ping()

            logger.info("Cache manager initialized", redis_url=self.redis_url)

        except Exception as e:
            logger.error("Failed to initialize cache manager", error=str(e))
            # Continue without caching if Redis is unavailable
            self.redis = None

    async def get(self, key: str, default: Any = None, deserialize: bool = True) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found
            deserialize: Whether to deserialize the value

        Returns:
            Cached value or default
        """
        if not self.redis:
            self._cache_stats["misses"] += 1
            return default

        full_key = self._make_key(key)

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                value = await self.redis.get(full_key)

                # Record metrics
                duration = time.time() - start_time
                if value is not None:
                    self._cache_stats["hits"] += 1
                    logger.debug("Cache hit", key=key, duration=duration)
                else:
                    self._cache_stats["misses"] += 1
                    logger.debug("Cache miss", key=key)

                if value is None:
                    return default

                if deserialize:
                    return self._deserialize(value)
                else:
                    return value.decode("utf-8") if isinstance(value, bytes) else value

            except Exception as e:
                self._cache_stats["errors"] += 1
                logger.warning(
                    "Cache get error", key=key, attempt=attempt + 1, error=str(e)
                )

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                else:
                    return default

        return default

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None, serialize: bool = True
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize the value

        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False

        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                if serialize:
                    serialized_value = self._serialize(value)
                else:
                    serialized_value = (
                        value.encode("utf-8") if isinstance(value, str) else value
                    )

                await self.redis.set(full_key, serialized_value, ex=ttl)

                duration = time.time() - start_time
                self._cache_stats["sets"] += 1

                logger.debug("Cache set", key=key, ttl=ttl, duration=duration)
                return True

            except Exception as e:
                self._cache_stats["errors"] += 1
                logger.warning(
                    "Cache set error", key=key, attempt=attempt + 1, error=str(e)
                )

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        if not self.redis:
            return False

        full_key = self._make_key(key)

        try:
            result = await self.redis.delete(full_key)
            if result > 0:
                self._cache_stats["deletes"] += 1
                logger.debug("Cache delete", key=key)
                return True
            return False

        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.warning("Cache delete error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis:
            return False

        full_key = self._make_key(key)

        try:
            result = await self.redis.exists(full_key)
            return result > 0

        except Exception as e:
            logger.warning("Cache exists check error", key=key, error=str(e))
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key."""
        if not self.redis:
            return False

        full_key = self._make_key(key)

        try:
            result = await self.redis.expire(full_key, ttl)
            return result

        except Exception as e:
            logger.warning("Cache expire error", key=key, error=str(e))
            return False

    async def increment(
        self, key: str, amount: int = 1, ttl: Optional[int] = None
    ) -> Optional[int]:
        """Increment counter in cache."""
        if not self.redis:
            return None

        full_key = self._make_key(key)

        try:
            # Use pipeline for atomic increment + expire
            async with self.redis.pipeline() as pipe:
                pipe.incr(full_key, amount)
                if ttl:
                    pipe.expire(full_key, ttl)
                results = await pipe.execute()
                return results[0]

        except Exception as e:
            logger.warning("Cache increment error", key=key, error=str(e))
            return None

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.

        Args:
            pattern: Pattern to match (use * for wildcard)

        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0

        full_pattern = self._make_key(pattern)

        try:
            # Scan for keys matching pattern
            keys = []
            async for key in self.redis.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info("Cache pattern clear", pattern=pattern, deleted=deleted)
                return deleted

            return 0

        except Exception as e:
            logger.warning("Cache pattern clear error", pattern=pattern, error=str(e))
            return 0

    async def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        stats = dict(self._cache_stats)

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0

        # Get Redis info if available
        if self.redis:
            try:
                redis_info = await self.redis.info()
                stats.update(
                    {
                        "redis_connected_clients": redis_info.get(
                            "connected_clients", 0
                        ),
                        "redis_used_memory": redis_info.get("used_memory", 0),
                        "redis_keyspace_hits": redis_info.get("keyspace_hits", 0),
                        "redis_keyspace_misses": redis_info.get("keyspace_misses", 0),
                    }
                )
            except Exception as e:
                logger.warning("Failed to get Redis info", error=str(e))

        return stats

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                return json.dumps(value).encode("utf-8")
            else:
                # Fall back to pickle for complex objects
                return b"PICKLE:" + pickle.dumps(value)
        except Exception:
            # Final fallback - pickle everything
            return b"PICKLE:" + pickle.dumps(value)

    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from storage."""
        if value.startswith(b"PICKLE:"):
            return pickle.loads(value[7:])  # Remove "PICKLE:" prefix
        else:
            try:
                return json.loads(value.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to pickle if JSON fails
                return pickle.loads(value)

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Cache manager closed")


# Specific cache instances for different use cases
class ResponseCache:
    """Specialized cache for response data."""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    async def get_response(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        return await self.cache.get(f"response:{response_id}")

    async def set_response(
        self, response_id: str, response_data: Dict[str, Any], ttl: int = 3600
    ) -> bool:
        """Cache response data."""
        return await self.cache.set(f"response:{response_id}", response_data, ttl=ttl)

    async def get_context_chain(self, response_id: str) -> Optional[list]:
        """Get cached conversation context chain."""
        return await self.cache.get(f"context_chain:{response_id}")

    async def set_context_chain(
        self, response_id: str, chain: list, ttl: int = 1800
    ) -> bool:
        """Cache conversation context chain."""
        return await self.cache.set(f"context_chain:{response_id}", chain, ttl=ttl)


class EmbeddingCache:
    """Specialized cache for embeddings."""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    async def get_embedding(self, text_hash: str) -> Optional[list]:
        """Get cached embedding."""
        return await self.cache.get(f"embedding:{text_hash}")

    async def set_embedding(
        self, text_hash: str, embedding: list, ttl: int = 86400
    ) -> bool:
        """Cache embedding (24 hour TTL)."""
        return await self.cache.set(f"embedding:{text_hash}", embedding, ttl=ttl)

    async def get_search_results(
        self, query_hash: str, vector_store_id: str
    ) -> Optional[list]:
        """Get cached search results."""
        return await self.cache.get(f"search:{vector_store_id}:{query_hash}")

    async def set_search_results(
        self, query_hash: str, vector_store_id: str, results: list, ttl: int = 300
    ) -> bool:
        """Cache search results (5 minute TTL)."""
        return await self.cache.set(
            f"search:{vector_store_id}:{query_hash}", results, ttl=ttl
        )
