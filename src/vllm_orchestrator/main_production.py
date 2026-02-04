"""Production-ready FastAPI application for vLLM Orchestrator."""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vllm_orchestrator.config import Settings
from vllm_orchestrator.middleware.rate_limiting import (
    RateLimitConfig,
    RateLimitMiddleware,
)
from vllm_orchestrator.middleware.security import SecurityConfig, SecurityMiddleware
from vllm_orchestrator.observability.logging import (
    RequestIDMiddleware,
    configure_logging,
    get_logger,
)
from vllm_orchestrator.observability.metrics import (
    MetricsMiddleware,
    metrics_endpoint,
    setup_metrics,
)
from vllm_orchestrator.performance.caching import CacheManager
from vllm_orchestrator.performance.connection_pool import (
    ConnectionPoolMonitor,
    DatabaseConnectionPool,
)

logger = get_logger(__name__)


class ApplicationState:
    """Global application state management."""

    def __init__(self):
        self.settings: Settings = Settings()
        self.db_pool: DatabaseConnectionPool = None
        self.cache_manager: CacheManager = None
        self.pool_monitor: ConnectionPoolMonitor = None
        self.shutdown_event: asyncio.Event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize application components."""
        logger.info("Initializing application components")

        # Initialize database connection pool
        self.db_pool = DatabaseConnectionPool(
            database_url=self.settings.database.url,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_timeout=self.settings.database.pool_timeout,
            echo_sql=self.settings.database.echo_sql,
        )
        await self.db_pool.initialize()

        # Initialize cache manager
        if self.settings.redis.url:
            self.cache_manager = CacheManager(
                redis_url=self.settings.redis.url,
                default_ttl=self.settings.redis.default_ttl,
            )
            await self.cache_manager.initialize()

        # Start connection pool monitoring
        if self.settings.monitoring.enable_db_monitoring:
            self.pool_monitor = ConnectionPoolMonitor(
                self.db_pool, check_interval=self.settings.monitoring.db_check_interval
            )
            await self.pool_monitor.start_monitoring()

        # Set up metrics
        setup_metrics("vllm-orchestrator", "1.0.0")

        logger.info("Application initialization complete")

    async def shutdown(self) -> None:
        """Shutdown application components."""
        logger.info("Shutting down application components")

        # Stop monitoring
        if self.pool_monitor:
            await self.pool_monitor.stop_monitoring()

        # Close cache manager
        if self.cache_manager:
            await self.cache_manager.close()

        # Close database pool
        if self.db_pool:
            await self.db_pool.close()

        self.shutdown_event.set()
        logger.info("Application shutdown complete")


# Global application state
app_state = ApplicationState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Configure logging first
    configure_logging(
        level=app_state.settings.logging.level,
        json_logs=app_state.settings.logging.json_logs,
        enable_access_logs=app_state.settings.logging.enable_access_logs,
    )

    # Initialize application
    await app_state.initialize()

    # Set up graceful shutdown handling
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(app_state.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    yield

    # Cleanup on shutdown
    if not app_state.shutdown_event.is_set():
        await app_state.shutdown()


def create_app() -> FastAPI:
    """Create FastAPI application with all middleware and routes."""
    app = FastAPI(
        title="vLLM Orchestrator",
        description="Stateful response orchestration for vLLM",
        version="1.0.0",
        docs_url="/docs" if app_state.settings.server.enable_docs else None,
        redoc_url="/redoc" if app_state.settings.server.enable_docs else None,
        lifespan=lifespan,
    )

    # Add middleware (order matters - last added runs first)

    # 1. Request ID middleware (runs first)
    app.add_middleware(RequestIDMiddleware)

    # 2. Security middleware
    if app_state.settings.security.enable_security_headers:
        security_config = SecurityConfig(
            allow_origins=app_state.settings.security.cors_origins,
            require_api_key=app_state.settings.security.require_api_key,
            api_keys=set(app_state.settings.security.api_keys)
            if app_state.settings.security.api_keys
            else set(),
            max_request_size=app_state.settings.security.max_request_size,
        )
        app.add_middleware(SecurityMiddleware, config=security_config)

    # 3. Rate limiting middleware
    if (
        app_state.settings.rate_limiting.enable_rate_limiting
        and app_state.settings.redis.url
    ):
        rate_limit_config = RateLimitConfig(
            global_requests_per_minute=app_state.settings.rate_limiting.global_requests_per_minute,
            tenant_requests_per_minute=app_state.settings.rate_limiting.tenant_requests_per_minute,
            responses_per_minute=app_state.settings.rate_limiting.responses_per_minute,
        )
        app.add_middleware(
            RateLimitMiddleware,
            redis_url=app_state.settings.redis.url,
            config=rate_limit_config,
        )

    # 4. Metrics middleware
    if app_state.settings.monitoring.enable_metrics:
        app.add_middleware(MetricsMiddleware)

    # 5. CORS middleware (if security middleware doesn't handle it)
    if not app_state.settings.security.enable_security_headers:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_state.settings.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add routes
    add_routes(app)

    # Add exception handlers
    add_exception_handlers(app)

    return app


def add_routes(app: FastAPI) -> None:
    """Add all application routes."""

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "version": "1.0.0",
            "components": {},
        }

        # Check database
        if app_state.db_pool:
            db_healthy = await app_state.db_pool.health_check()
            health_status["components"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "stats": app_state.db_pool.get_stats(),
            }

        # Check cache
        if app_state.cache_manager:
            try:
                cache_stats = await app_state.cache_manager.get_stats()
                health_status["components"]["cache"] = {
                    "status": "healthy",
                    "stats": cache_stats,
                }
            except Exception as e:
                health_status["components"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        # Overall status
        component_statuses = [
            comp.get("status", "unknown")
            for comp in health_status["components"].values()
        ]
        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "degraded"

        return health_status

    # Metrics endpoint
    if app_state.settings.monitoring.enable_metrics:
        app.get("/metrics")(metrics_endpoint)

    # Readiness check (for Kubernetes)
    @app.get("/ready")
    async def readiness_check() -> Dict[str, str]:
        """Readiness check for container orchestration."""
        if app_state.shutdown_event.is_set():
            return JSONResponse(
                content={"status": "not_ready", "reason": "shutting_down"},
                status_code=503,
            )

        # Check critical components
        if app_state.db_pool:
            db_healthy = await app_state.db_pool.health_check()
            if not db_healthy:
                return JSONResponse(
                    content={"status": "not_ready", "reason": "database_unavailable"},
                    status_code=503,
                )

        return {"status": "ready"}

    # Liveness check (for Kubernetes)
    @app.get("/live")
    async def liveness_check() -> Dict[str, str]:
        """Liveness check for container orchestration."""
        if app_state.shutdown_event.is_set():
            return JSONResponse(
                content={"status": "not_alive", "reason": "shutting_down"},
                status_code=503,
            )

        return {"status": "alive"}

    # TODO: Add API routes here
    # from vllm_orchestrator.api.responses import router as responses_router
    # from vllm_orchestrator.api.files import router as files_router
    # from vllm_orchestrator.api.vector_stores import router as vector_stores_router

    # app.include_router(responses_router, prefix="/v1")
    # app.include_router(files_router, prefix="/v1")
    # app.include_router(vector_stores_router, prefix="/v1")


def add_exception_handlers(app: FastAPI) -> None:
    """Add global exception handlers."""

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Global exception handler for unhandled errors."""
        logger.error(
            "Unhandled exception",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            path=request.url.path,
            method=request.method,
            client_ip=request.client.host if request.client else "unknown",
            exc_info=exc,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request.headers.get("x-request-id"),
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> JSONResponse:
        """404 handler."""
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not found",
                "message": f"Endpoint {request.url.path} not found",
                "request_id": request.headers.get("x-request-id"),
            },
        )


# Create the FastAPI app
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Get settings for development server
    settings = Settings()

    uvicorn.run(
        "vllm_orchestrator.main_production:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level="info",
        access_log=True,
        workers=1 if settings.server.debug else settings.server.workers,
    )
