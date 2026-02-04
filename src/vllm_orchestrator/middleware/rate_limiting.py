"""Rate limiting middleware using Redis."""

import asyncio
import time
from typing import Dict, Optional, Tuple

import redis.asyncio as aioredis
from fastapi import HTTPException, status
from starlette.requests import Request
from starlette.responses import Response

from vllm_orchestrator.observability.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def is_allowed(
        self, key: str, limit: int, window_seconds: int = 60
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed under rate limit.

        Uses sliding window algorithm with Redis sorted sets.

        Args:
            key: Rate limit key (e.g., "user:123" or "tenant:abc")
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, rate_limit_info)
            rate_limit_info contains: remaining, reset_time, retry_after
        """
        current_time = int(time.time())
        window_start = current_time - window_seconds

        # Use Redis pipeline for atomic operations
        async with self.redis.pipeline() as pipe:
            try:
                # Remove expired entries
                pipe.zremrangebyscore(key, 0, window_start)

                # Count current requests in window
                pipe.zcard(key)

                # Add current request (tentatively)
                pipe.zadd(key, {str(current_time): current_time})

                # Set expiration on the key
                pipe.expire(key, window_seconds + 1)

                results = await pipe.execute()
                current_count = results[1]

                if current_count < limit:
                    # Request allowed
                    remaining = limit - current_count - 1
                    reset_time = current_time + window_seconds

                    return True, {
                        "remaining": remaining,
                        "reset_time": reset_time,
                        "retry_after": 0,
                    }
                else:
                    # Rate limit exceeded - remove the tentative request
                    await self.redis.zrem(key, str(current_time))

                    # Calculate retry after time
                    oldest_request = await self.redis.zrange(key, 0, 0, withscores=True)
                    if oldest_request:
                        oldest_time = int(oldest_request[0][1])
                        retry_after = max(
                            0, oldest_time + window_seconds - current_time
                        )
                    else:
                        retry_after = window_seconds

                    return False, {
                        "remaining": 0,
                        "reset_time": current_time + window_seconds,
                        "retry_after": retry_after,
                    }

            except Exception as e:
                logger.error("Rate limit check failed", error=str(e), key=key)
                # Fail open - allow request if Redis is down
                return True, {
                    "remaining": limit - 1,
                    "reset_time": current_time + window_seconds,
                    "retry_after": 0,
                }


class RateLimitConfig:
    """Rate limiting configuration."""

    def __init__(
        self,
        *,
        # Global rate limits
        global_requests_per_minute: int = 1000,
        global_requests_per_hour: int = 10000,
        # Per-tenant rate limits
        tenant_requests_per_minute: int = 100,
        tenant_requests_per_hour: int = 1000,
        # Per-user rate limits (if user auth is implemented)
        user_requests_per_minute: int = 20,
        user_requests_per_hour: int = 200,
        # Response generation specific limits
        responses_per_minute: int = 10,
        responses_per_hour: int = 100,
        # Tool execution limits
        tool_calls_per_minute: int = 50,
        tool_calls_per_hour: int = 500,
        # Vector search limits
        vector_searches_per_minute: int = 30,
        vector_searches_per_hour: int = 300,
    ):
        self.global_requests_per_minute = global_requests_per_minute
        self.global_requests_per_hour = global_requests_per_hour
        self.tenant_requests_per_minute = tenant_requests_per_minute
        self.tenant_requests_per_hour = tenant_requests_per_hour
        self.user_requests_per_minute = user_requests_per_minute
        self.user_requests_per_hour = user_requests_per_hour
        self.responses_per_minute = responses_per_minute
        self.responses_per_hour = responses_per_hour
        self.tool_calls_per_minute = tool_calls_per_minute
        self.tool_calls_per_hour = tool_calls_per_hour
        self.vector_searches_per_minute = vector_searches_per_minute
        self.vector_searches_per_hour = vector_searches_per_hour


class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI."""

    def __init__(
        self,
        app,
        redis_url: str = "redis://localhost:6379",
        config: Optional[RateLimitConfig] = None,
    ):
        self.app = app
        self.config = config or RateLimitConfig()
        self.redis_client: Optional[aioredis.Redis] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.redis_url = redis_url

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Initialize Redis connection if not already done
            if self.redis_client is None:
                try:
                    self.redis_client = aioredis.from_url(self.redis_url)
                    self.rate_limiter = RateLimiter(self.redis_client)
                    logger.info("Rate limiter initialized", redis_url=self.redis_url)
                except Exception as e:
                    logger.error("Failed to connect to Redis", error=str(e))
                    # Continue without rate limiting if Redis is unavailable
                    await self.app(scope, receive, send)
                    return

            request = Request(scope, receive)

            # Extract tenant ID and user ID for rate limiting
            tenant_id = self._extract_tenant_id(request)
            user_id = self._extract_user_id(request)

            # Apply rate limits
            try:
                rate_limit_exceeded = await self._check_rate_limits(
                    request, tenant_id, user_id
                )

                if rate_limit_exceeded:
                    # Return 429 Too Many Requests
                    response = Response(
                        content="Rate limit exceeded",
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        headers={
                            "Retry-After": str(rate_limit_exceeded["retry_after"]),
                            "X-RateLimit-Limit": str(
                                rate_limit_exceeded.get("limit", 0)
                            ),
                            "X-RateLimit-Remaining": str(
                                rate_limit_exceeded["remaining"]
                            ),
                            "X-RateLimit-Reset": str(rate_limit_exceeded["reset_time"]),
                        },
                    )

                    await response(scope, receive, send)
                    return

            except Exception as e:
                logger.error("Rate limiting error", error=str(e))
                # Continue without rate limiting on error

        await self.app(scope, receive, send)

    async def _check_rate_limits(
        self, request: Request, tenant_id: str, user_id: Optional[str]
    ) -> Optional[Dict[str, int]]:
        """Check various rate limits and return exceeded info if any."""
        path = request.url.path
        method = request.method

        # Global rate limits
        is_allowed, info = await self.rate_limiter.is_allowed(
            f"global:requests", self.config.global_requests_per_minute, 60
        )
        if not is_allowed:
            return {**info, "limit": self.config.global_requests_per_minute}

        is_allowed, info = await self.rate_limiter.is_allowed(
            f"global:requests", self.config.global_requests_per_hour, 3600
        )
        if not is_allowed:
            return {**info, "limit": self.config.global_requests_per_hour}

        # Tenant-specific rate limits
        is_allowed, info = await self.rate_limiter.is_allowed(
            f"tenant:{tenant_id}:requests", self.config.tenant_requests_per_minute, 60
        )
        if not is_allowed:
            return {**info, "limit": self.config.tenant_requests_per_minute}

        is_allowed, info = await self.rate_limiter.is_allowed(
            f"tenant:{tenant_id}:requests", self.config.tenant_requests_per_hour, 3600
        )
        if not is_allowed:
            return {**info, "limit": self.config.tenant_requests_per_hour}

        # User-specific rate limits (if user is identified)
        if user_id:
            is_allowed, info = await self.rate_limiter.is_allowed(
                f"user:{user_id}:requests", self.config.user_requests_per_minute, 60
            )
            if not is_allowed:
                return {**info, "limit": self.config.user_requests_per_minute}

        # Endpoint-specific rate limits
        if path.startswith("/v1/responses"):
            is_allowed, info = await self.rate_limiter.is_allowed(
                f"tenant:{tenant_id}:responses", self.config.responses_per_minute, 60
            )
            if not is_allowed:
                return {**info, "limit": self.config.responses_per_minute}

        elif path.startswith("/v1/vector_stores") and "search" in path:
            is_allowed, info = await self.rate_limiter.is_allowed(
                f"tenant:{tenant_id}:vector_searches",
                self.config.vector_searches_per_minute,
                60,
            )
            if not is_allowed:
                return {**info, "limit": self.config.vector_searches_per_minute}

        return None

    def _extract_tenant_id(self, request: Request) -> str:
        """Extract tenant ID from request."""
        # Check headers first
        tenant_id = request.headers.get("x-tenant-id")
        if tenant_id:
            return tenant_id

        # Check query parameters
        tenant_id = request.query_params.get("tenant_id")
        if tenant_id:
            return tenant_id

        # Check authorization header for tenant info
        auth_header = request.headers.get("authorization")
        if auth_header:
            # Parse JWT or API key to extract tenant
            # This would be implemented based on your auth scheme
            pass

        # Default tenant
        return "default"

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available."""
        # Check headers
        user_id = request.headers.get("x-user-id")
        if user_id:
            return user_id

        # Parse from authorization token
        auth_header = request.headers.get("authorization")
        if auth_header:
            # Parse JWT or API key to extract user
            # This would be implemented based on your auth scheme
            pass

        return None
