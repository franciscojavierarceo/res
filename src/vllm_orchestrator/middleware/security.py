"""Security middleware for production deployments."""

import secrets
import time
from typing import Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from vllm_orchestrator.observability.logging import get_logger

logger = get_logger(__name__)


class SecurityConfig:
    """Security configuration."""

    def __init__(
        self,
        *,
        # CORS settings
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        expose_headers: List[str] = None,
        allow_credentials: bool = False,
        max_age: int = 86400,
        # Security headers
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        enable_csp: bool = True,
        csp_policy: Optional[str] = None,
        enable_x_frame_options: bool = True,
        x_frame_options: str = "DENY",
        # API key authentication
        api_keys: Optional[Set[str]] = None,
        require_api_key: bool = False,
        api_key_header: str = "x-api-key",
        # Request validation
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        max_path_length: int = 2048,
        max_header_size: int = 8192,
        blocked_user_agents: Optional[List[str]] = None,
        blocked_ips: Optional[Set[str]] = None,
        # Rate limiting for security
        max_requests_per_ip: int = 1000,
        rate_limit_window: int = 3600,  # 1 hour
    ):
        # CORS
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
        ]
        self.allow_headers = allow_headers or ["*"]
        self.expose_headers = expose_headers or []
        self.allow_credentials = allow_credentials
        self.max_age = max_age

        # Security headers
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.enable_csp = enable_csp
        self.csp_policy = csp_policy or (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        self.enable_x_frame_options = enable_x_frame_options
        self.x_frame_options = x_frame_options

        # Authentication
        self.api_keys = api_keys or set()
        self.require_api_key = require_api_key
        self.api_key_header = api_key_header

        # Request validation
        self.max_request_size = max_request_size
        self.max_path_length = max_path_length
        self.max_header_size = max_header_size
        self.blocked_user_agents = [ua.lower() for ua in (blocked_user_agents or [])]
        self.blocked_ips = blocked_ips or set()

        # Rate limiting
        self.max_requests_per_ip = max_requests_per_ip
        self.rate_limit_window = rate_limit_window


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with CORS, headers, and basic auth."""

    def __init__(self, app, config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.config = config or SecurityConfig()
        self.ip_request_counts: Dict[str, Dict[str, int]] = {}

    async def dispatch(self, request: Request, call_next):
        """Process request through security checks."""
        client_ip = self._get_client_ip(request)

        # Security validations
        security_response = await self._validate_request_security(request, client_ip)
        if security_response:
            return security_response

        # Handle CORS preflight
        if request.method == "OPTIONS":
            return self._create_cors_response()

        # API key authentication
        if self.config.require_api_key:
            auth_response = self._validate_api_key(request)
            if auth_response:
                return auth_response

        # Track IP requests for basic rate limiting
        self._track_ip_request(client_ip)

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error("Request processing failed", error=str(e), client_ip=client_ip)
            raise

        # Add security headers
        self._add_security_headers(response)

        # Add CORS headers
        self._add_cors_headers(response, request)

        return response

    async def _validate_request_security(
        self, request: Request, client_ip: str
    ) -> Optional[Response]:
        """Validate request against security policies."""

        # Check blocked IPs
        if client_ip in self.config.blocked_ips:
            logger.warning("Blocked IP attempted access", client_ip=client_ip)
            return Response("Forbidden", status_code=403)

        # Check path length
        if len(request.url.path) > self.config.max_path_length:
            logger.warning(
                "Path too long", path_length=len(request.url.path), client_ip=client_ip
            )
            return Response("URI Too Long", status_code=414)

        # Check header size
        total_header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_size > self.config.max_header_size:
            logger.warning(
                "Headers too large", header_size=total_header_size, client_ip=client_ip
            )
            return Response("Request Header Fields Too Large", status_code=431)

        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_request_size:
            logger.warning(
                "Request too large", content_length=content_length, client_ip=client_ip
            )
            return Response("Payload Too Large", status_code=413)

        # Check user agent
        user_agent = request.headers.get("user-agent", "").lower()
        for blocked_ua in self.config.blocked_user_agents:
            if blocked_ua in user_agent:
                logger.warning(
                    "Blocked user agent", user_agent=user_agent, client_ip=client_ip
                )
                return Response("Forbidden", status_code=403)

        # Basic IP rate limiting
        if self._is_ip_rate_limited(client_ip):
            logger.warning("IP rate limited", client_ip=client_ip)
            return Response(
                "Too Many Requests",
                status_code=429,
                headers={"Retry-After": str(self.config.rate_limit_window)},
            )

        return None

    def _validate_api_key(self, request: Request) -> Optional[Response]:
        """Validate API key if required."""
        api_key = request.headers.get(self.config.api_key_header)

        if not api_key:
            logger.warning("Missing API key", path=request.url.path)
            return Response(
                "API key required",
                status_code=401,
                headers={
                    "WWW-Authenticate": f'ApiKey realm="{self.config.api_key_header}"'
                },
            )

        if api_key not in self.config.api_keys:
            logger.warning(
                "Invalid API key", api_key_prefix=api_key[:8] if api_key else ""
            )
            return Response("Invalid API key", status_code=401)

        return None

    def _track_ip_request(self, client_ip: str) -> None:
        """Track requests per IP for basic rate limiting."""
        current_time = int(time.time())
        window_start = current_time - self.config.rate_limit_window

        # Clean old entries
        for ip in list(self.ip_request_counts.keys()):
            self.ip_request_counts[ip] = {
                timestamp: count
                for timestamp, count in self.ip_request_counts[ip].items()
                if int(timestamp) > window_start
            }
            if not self.ip_request_counts[ip]:
                del self.ip_request_counts[ip]

        # Track current request
        if client_ip not in self.ip_request_counts:
            self.ip_request_counts[client_ip] = {}

        timestamp_key = str(current_time // 60)  # Group by minute
        self.ip_request_counts[client_ip][timestamp_key] = (
            self.ip_request_counts[client_ip].get(timestamp_key, 0) + 1
        )

    def _is_ip_rate_limited(self, client_ip: str) -> bool:
        """Check if IP is rate limited."""
        if client_ip not in self.ip_request_counts:
            return False

        total_requests = sum(self.ip_request_counts[client_ip].values())
        return total_requests > self.config.max_requests_per_ip

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP considering proxies."""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _create_cors_response(self) -> Response:
        """Create CORS preflight response."""
        response = Response(status_code=200)
        self._add_cors_headers(response, None)
        return response

    def _add_cors_headers(self, response: Response, request: Optional[Request]) -> None:
        """Add CORS headers to response."""
        # Access-Control-Allow-Origin
        if request:
            origin = request.headers.get("origin")
            if origin and (
                "*" in self.config.allow_origins or origin in self.config.allow_origins
            ):
                response.headers["Access-Control-Allow-Origin"] = origin
            elif "*" in self.config.allow_origins:
                response.headers["Access-Control-Allow-Origin"] = "*"
        elif "*" in self.config.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"

        # Other CORS headers
        if self.config.allow_methods:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(
                self.config.allow_methods
            )

        if self.config.allow_headers:
            response.headers["Access-Control-Allow-Headers"] = ", ".join(
                self.config.allow_headers
            )

        if self.config.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(
                self.config.expose_headers
            )

        if self.config.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        response.headers["Access-Control-Max-Age"] = str(self.config.max_age)

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        # HSTS
        if self.config.enable_hsts:
            hsts_value = f"max-age={self.config.hsts_max_age}"
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Content Security Policy
        if self.config.enable_csp and self.config.csp_policy:
            response.headers["Content-Security-Policy"] = self.config.csp_policy

        # X-Frame-Options
        if self.config.enable_x_frame_options:
            response.headers["X-Frame-Options"] = self.config.x_frame_options

        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=(), "
            "speaker=(), fullscreen=(self)"
        )
