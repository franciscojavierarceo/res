"""Database connection management with provider-agnostic factory pattern."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vllm_orchestrator.config import (
    PostgresStorageConfig,
    SqliteStorageConfig,
    StackConfig,
    StorageBackendConfig,
)
from vllm_orchestrator.storage.models import Base


class DatabaseManager:
    """Manages database connections and sessions.

    Provider-agnostic: works with SQLite, PostgreSQL, or any SQLAlchemy-supported backend.
    """

    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        self._session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def create_tables(self) -> None:
        """Create all tables defined in models."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables. Use with caution."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup."""
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def close(self) -> None:
        """Close the database engine."""
        await self._engine.dispose()


def _create_engine_for_sqlite(config: SqliteStorageConfig) -> AsyncEngine:
    """Create SQLAlchemy engine for SQLite."""
    # SQLite needs special handling for async
    url = f"sqlite+aiosqlite:///{config.db_path}"
    return create_async_engine(
        url,
        echo=False,
        # SQLite doesn't support pool settings the same way
        connect_args={"check_same_thread": False},
    )


def _create_engine_for_postgres(config: PostgresStorageConfig) -> AsyncEngine:
    """Create SQLAlchemy engine for PostgreSQL."""
    return create_async_engine(
        config.connection_url,
        echo=False,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_pre_ping=True,  # Verify connections before use
    )


def create_engine_from_config(config: StorageBackendConfig) -> AsyncEngine:
    """Factory function: create appropriate engine based on config type.

    Uses the discriminated union pattern - config.type determines which
    implementation to use.
    """
    match config:
        case SqliteStorageConfig():
            return _create_engine_for_sqlite(config)
        case PostgresStorageConfig():
            return _create_engine_for_postgres(config)
        case _:
            raise ValueError(f"Unknown storage backend type: {type(config)}")


def create_database_manager(config: StorageBackendConfig) -> DatabaseManager:
    """Create a DatabaseManager from storage backend config."""
    engine = create_engine_from_config(config)
    return DatabaseManager(engine)


# -----------------------------------------------------------------------------
# Global instance management (set during app startup)
# -----------------------------------------------------------------------------

_database_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance.

    Raises RuntimeError if called before initialization.
    """
    if _database_manager is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() during app startup."
        )
    return _database_manager


async def init_database(
    config: StackConfig, backend_name: str = "default"
) -> DatabaseManager:
    """Initialize the global database manager.

    Args:
        config: Stack configuration
        backend_name: Name of the storage backend to use from config.storage_backends

    Returns:
        The initialized DatabaseManager
    """
    global _database_manager

    if backend_name not in config.storage_backends:
        raise ValueError(
            f"Storage backend '{backend_name}' not found in config. "
            f"Available: {list(config.storage_backends.keys())}"
        )

    backend_config = config.storage_backends[backend_name]
    _database_manager = create_database_manager(backend_config)
    await _database_manager.create_tables()

    return _database_manager


async def close_database() -> None:
    """Close the global database manager."""
    global _database_manager
    if _database_manager is not None:
        await _database_manager.close()
        _database_manager = None
