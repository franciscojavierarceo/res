"""Optimized database connection pooling."""

import asyncio
import time
from typing import AsyncContextManager, Dict, Optional

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from vllm_orchestrator.observability.logging import get_logger
from vllm_orchestrator.observability.metrics import metrics_registry

logger = get_logger(__name__)


class DatabaseConnectionPool:
    """Optimized database connection pool with monitoring."""

    def __init__(
        self,
        database_url: str,
        *,
        # Connection pool settings
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,  # 1 hour
        pool_pre_ping: bool = True,
        # Performance settings
        echo_sql: bool = False,
        statement_timeout: int = 30000,  # 30 seconds
        query_timeout: int = 60000,  # 60 seconds
        # Connection settings
        connect_timeout: int = 10,
        server_side_cursors: bool = True,
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.echo_sql = echo_sql
        self.statement_timeout = statement_timeout
        self.query_timeout = query_timeout
        self.connect_timeout = connect_timeout
        self.server_side_cursors = server_side_cursors

        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._connection_stats: Dict[str, int] = {
            "total_connections": 0,
            "active_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            # Configure connection parameters
            connect_args = {
                "connect_timeout": self.connect_timeout,
                "statement_timeout": self.statement_timeout,
                "query_timeout": self.query_timeout,
            }

            # Use QueuePool for PostgreSQL, NullPool for SQLite
            if "postgresql" in self.database_url:
                poolclass = QueuePool
                connect_args.update(
                    {
                        "server_side_cursors": self.server_side_cursors,
                        "prepared_statement_cache_size": 100,
                        "prepared_statement_name_func": lambda: f"stmt_{int(time.time() * 1000000) % 1000000}",
                    }
                )
            else:
                poolclass = NullPool  # SQLite doesn't support connection pooling

            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                poolclass=poolclass,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
                echo=self.echo_sql,
                connect_args=connect_args,
                # Performance optimizations
                execution_options={
                    "compiled_cache": {},
                    "isolation_level": "AUTOCOMMIT",
                },
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=False,  # Manual control over flushing
                autocommit=False,
            )

            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")

            logger.info(
                "Database connection pool initialized",
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                database_type="postgresql"
                if "postgresql" in self.database_url
                else "sqlite",
            )

        except Exception as e:
            logger.error("Failed to initialize database connection pool", error=str(e))
            raise

    async def get_session(self) -> AsyncContextManager:
        """Get a database session with automatic cleanup."""
        if not self.session_factory:
            raise RuntimeError("Connection pool not initialized")

        session = self.session_factory()
        start_time = time.time()

        try:
            self._connection_stats["active_connections"] += 1
            self._connection_stats["total_connections"] += 1

            # Record database connection metric
            metrics_registry.database_operations_total.labels(
                operation="connect", table="session", status="success"
            ).inc()

            class SessionContextManager:
                def __init__(self, session, stats):
                    self._session = session
                    self._stats = stats

                async def __aenter__(self):
                    return self._session

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    try:
                        if exc_type:
                            await self._session.rollback()
                            logger.warning(
                                "Database session rolled back",
                                exception_type=exc_type.__name__ if exc_type else None,
                            )
                        else:
                            await self._session.commit()

                        await self._session.close()

                    except Exception as e:
                        logger.error("Error closing database session", error=str(e))
                    finally:
                        self._stats["active_connections"] -= 1

                        # Record session duration
                        duration = time.time() - start_time
                        metrics_registry.database_operation_duration_seconds.labels(
                            operation="session", table="session"
                        ).observe(duration)

            return SessionContextManager(session, self._connection_stats)

        except Exception as e:
            self._connection_stats["active_connections"] -= 1
            await session.close()

            metrics_registry.database_operations_total.labels(
                operation="connect", table="session", status="failure"
            ).inc()

            logger.error("Failed to create database session", error=str(e))
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection pool closed")

    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        pool_stats = {
            **self._connection_stats,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
        }

        # Add SQLAlchemy pool stats if available
        if self.engine and hasattr(self.engine.pool, "size"):
            pool_stats.update(
                {
                    "pool_current_size": self.engine.pool.size(),
                    "pool_checked_in": self.engine.pool.checkedin(),
                    "pool_checked_out": self.engine.pool.checkedout(),
                    "pool_overflow": getattr(self.engine.pool, "overflow", lambda: 0)(),
                }
            )

        return pool_stats

    async def health_check(self) -> bool:
        """Check if the database connection pool is healthy."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False


class ConnectionPoolMonitor:
    """Monitor connection pool health and performance."""

    def __init__(self, pool: DatabaseConnectionPool, check_interval: int = 60):
        self.pool = pool
        self.check_interval = check_interval
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self) -> None:
        """Start connection pool monitoring."""
        if self._monitor_task:
            return

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Connection pool monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop connection pool monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Connection pool monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Monitor connection pool in a loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)

                # Get pool statistics
                stats = self.pool.get_stats()

                # Log statistics
                logger.info("Connection pool stats", **stats)

                # Check for issues
                active_ratio = stats["active_connections"] / max(stats["pool_size"], 1)
                if active_ratio > 0.8:
                    logger.warning(
                        "High connection pool usage",
                        active_connections=stats["active_connections"],
                        pool_size=stats["pool_size"],
                        usage_ratio=active_ratio,
                    )

                # Health check
                is_healthy = await self.pool.health_check()
                if not is_healthy:
                    logger.error("Database connection pool health check failed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Connection pool monitoring error", error=str(e))
