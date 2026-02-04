"""Configuration settings for vLLM Orchestrator using provider-agnostic patterns."""

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------------------------------------------------------------------
# Environment Variable Substitution (following Llama Stack pattern)
# -----------------------------------------------------------------------------

ENV_VAR_PATTERN = re.compile(r"\$\{env\.([A-Z_][A-Z0-9_]*)(?::=([^}]*))?\}")


def replace_env_vars(config: Any) -> Any:
    """Recursively replace ${env.VAR:=default} patterns in config."""
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_vars(v) for v in config]
    elif isinstance(config, str):

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            value = os.environ.get(var_name)
            if value is not None:
                return value
            if default is not None:
                return default
            raise ValueError(f"Environment variable {var_name} is required but not set")

        return ENV_VAR_PATTERN.sub(replacer, config)
    return config


# -----------------------------------------------------------------------------
# Storage Backend Configurations (Discriminated Union)
# -----------------------------------------------------------------------------


class SqliteStorageConfig(BaseModel):
    """SQLite storage backend configuration."""

    type: Literal["sqlite"] = "sqlite"
    db_path: str = Field(
        default="./orchestrator.db",
        description="File path for the SQLite database",
    )

    @classmethod
    def sample_config(cls, db_name: str = "orchestrator.db") -> dict[str, Any]:
        return {
            "type": "sqlite",
            "db_path": "${env.SQLITE_DB_PATH:=./" + db_name + "}",
        }


class PostgresStorageConfig(BaseModel):
    """PostgreSQL storage backend configuration."""

    type: Literal["postgres"] = "postgres"
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="orchestrator", description="Database name")
    user: str = Field(default="orchestrator", description="Database user")
    password: str | None = Field(default=None, description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")

    @property
    def connection_url(self) -> str:
        """Generate SQLAlchemy connection URL."""
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        return f"postgresql+asyncpg://{auth}{self.host}:{self.port}/{self.database}"

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "postgres",
            "host": "${env.POSTGRES_HOST:=localhost}",
            "port": "${env.POSTGRES_PORT:=5432}",
            "database": "${env.POSTGRES_DB:=orchestrator}",
            "user": "${env.POSTGRES_USER:=orchestrator}",
            "password": "${env.POSTGRES_PASSWORD}",
        }


# Discriminated union of storage backends
StorageBackendConfig = Annotated[
    SqliteStorageConfig | PostgresStorageConfig,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Vector Store Backend Configurations
# -----------------------------------------------------------------------------


class PgVectorConfig(BaseModel):
    """pgvector backend for vector storage."""

    type: Literal["pgvector"] = "pgvector"
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="orchestrator")
    user: str = Field(default="orchestrator")
    password: str | None = Field(default=None)
    distance_metric: Literal["cosine", "l2", "inner_product"] = Field(
        default="cosine",
        description="Distance metric for similarity search",
    )

    @property
    def connection_url(self) -> str:
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        return f"postgresql+asyncpg://{auth}{self.host}:{self.port}/{self.database}"

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "pgvector",
            "host": "${env.PGVECTOR_HOST:=localhost}",
            "port": "${env.PGVECTOR_PORT:=5432}",
            "database": "${env.PGVECTOR_DB:=orchestrator}",
            "user": "${env.PGVECTOR_USER:=orchestrator}",
            "password": "${env.PGVECTOR_PASSWORD}",
        }


class SqliteVectorConfig(BaseModel):
    """SQLite-based vector storage (for development only)."""

    type: Literal["sqlite_vec"] = "sqlite_vec"
    db_path: str = Field(
        default="./vectors.db",
        description="File path for the SQLite vector database",
    )

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "sqlite_vec",
            "db_path": "${env.SQLITE_VEC_PATH:=./vectors.db}",
        }


VectorBackendConfig = Annotated[
    PgVectorConfig | SqliteVectorConfig,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# File Storage Backend Configurations
# -----------------------------------------------------------------------------


class LocalFileStorageConfig(BaseModel):
    """Local filesystem storage for files."""

    type: Literal["local"] = "local"
    base_path: Path = Field(
        default=Path("./file_storage"),
        description="Base path for file storage",
    )
    max_file_size_mb: int = Field(
        default=512,
        description="Maximum file size in MB",
    )

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "local",
            "base_path": "${env.FILE_STORAGE_PATH:=./file_storage}",
            "max_file_size_mb": 512,
        }


class S3FileStorageConfig(BaseModel):
    """S3-compatible storage for files."""

    type: Literal["s3"] = "s3"
    bucket: str = Field(description="S3 bucket name")
    prefix: str = Field(default="", description="Key prefix for all files")
    region: str | None = Field(default=None, description="AWS region")
    endpoint_url: str | None = Field(
        default=None,
        description="Custom endpoint URL (for MinIO, etc.)",
    )
    access_key_id: str | None = Field(default=None)
    secret_access_key: str | None = Field(default=None)
    max_file_size_mb: int = Field(default=512)

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "s3",
            "bucket": "${env.S3_BUCKET}",
            "prefix": "${env.S3_PREFIX:=}",
            "region": "${env.AWS_REGION:=us-east-1}",
            "access_key_id": "${env.AWS_ACCESS_KEY_ID}",
            "secret_access_key": "${env.AWS_SECRET_ACCESS_KEY}",
        }


FileStorageBackendConfig = Annotated[
    LocalFileStorageConfig | S3FileStorageConfig,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Cache Backend Configurations
# -----------------------------------------------------------------------------


class MemoryCacheConfig(BaseModel):
    """In-memory cache (not distributed)."""

    type: Literal["memory"] = "memory"
    max_size: int = Field(default=10000, description="Maximum cache entries")
    ttl_seconds: int = Field(default=3600, description="Default TTL in seconds")


class RedisCacheConfig(BaseModel):
    """Redis cache backend."""

    type: Literal["redis"] = "redis"
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: str | None = Field(default=None)
    ttl_seconds: int = Field(default=3600)

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "redis",
            "host": "${env.REDIS_HOST:=localhost}",
            "port": "${env.REDIS_PORT:=6379}",
            "password": "${env.REDIS_PASSWORD:=}",
        }


CacheBackendConfig = Annotated[
    MemoryCacheConfig | RedisCacheConfig,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Inference Provider Configurations
# -----------------------------------------------------------------------------


class VllmInferenceConfig(BaseModel):
    """vLLM inference backend configuration."""

    type: Literal["vllm"] = "vllm"
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for vLLM server",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for vLLM server (if required)",
    )
    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds",
    )

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "vllm",
            "base_url": "${env.VLLM_BASE_URL:=http://localhost:8000}",
            "api_key": "${env.VLLM_API_KEY:=}",
            "timeout": 300.0,
        }


class OpenAIInferenceConfig(BaseModel):
    """OpenAI-compatible inference backend."""

    type: Literal["openai"] = "openai"
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible API",
    )
    api_key: str = Field(description="API key")
    timeout: float = Field(default=300.0)

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        return {
            "type": "openai",
            "base_url": "${env.OPENAI_BASE_URL:=https://api.openai.com/v1}",
            "api_key": "${env.OPENAI_API_KEY}",
        }


InferenceBackendConfig = Annotated[
    VllmInferenceConfig | OpenAIInferenceConfig,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Embedding Provider Configurations
# -----------------------------------------------------------------------------


class VllmEmbeddingConfig(BaseModel):
    """Use vLLM for embeddings."""

    type: Literal["vllm"] = "vllm"
    base_url: str = Field(default="http://localhost:8000")
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    dimension: int = Field(default=1536, description="Embedding dimension")


class OpenAIEmbeddingConfig(BaseModel):
    """Use OpenAI for embeddings."""

    type: Literal["openai"] = "openai"
    api_key: str = Field(description="OpenAI API key")
    model: str = Field(default="text-embedding-3-small")
    dimension: int = Field(default=1536)


EmbeddingBackendConfig = Annotated[
    VllmEmbeddingConfig | OpenAIEmbeddingConfig,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Storage References (following Llama Stack pattern)
# -----------------------------------------------------------------------------


class StorageReference(BaseModel):
    """Reference to a named storage backend."""

    backend: str = Field(description="Name of backend from storage.backends")


class StoresConfig(BaseModel):
    """Named store references."""

    responses: StorageReference = Field(
        default_factory=lambda: StorageReference(backend="default"),
    )
    files: StorageReference = Field(
        default_factory=lambda: StorageReference(backend="default"),
    )
    vectors: StorageReference = Field(
        default_factory=lambda: StorageReference(backend="vectors"),
    )


# -----------------------------------------------------------------------------
# Server Configuration
# -----------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    workers: int = Field(default=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


# -----------------------------------------------------------------------------
# Main Stack Configuration
# -----------------------------------------------------------------------------


class StackConfig(BaseModel):
    """Main configuration for vLLM Orchestrator stack."""

    version: int = Field(default=1, description="Config schema version")

    server: ServerConfig = Field(default_factory=ServerConfig)

    # Named backends
    storage_backends: dict[str, StorageBackendConfig] = Field(
        default_factory=lambda: {
            "default": SqliteStorageConfig(),
        },
        description="Named storage backends",
    )

    vector_backends: dict[str, VectorBackendConfig] = Field(
        default_factory=lambda: {
            "vectors": SqliteVectorConfig(),
        },
        description="Named vector storage backends",
    )

    file_backends: dict[str, FileStorageBackendConfig] = Field(
        default_factory=lambda: {
            "files": LocalFileStorageConfig(),
        },
        description="Named file storage backends",
    )

    cache_backends: dict[str, CacheBackendConfig] = Field(
        default_factory=lambda: {
            "cache": MemoryCacheConfig(),
        },
        description="Named cache backends",
    )

    # Provider configurations
    inference: InferenceBackendConfig = Field(
        default_factory=VllmInferenceConfig,
        description="Inference provider configuration",
    )

    embedding: EmbeddingBackendConfig = Field(
        default_factory=VllmEmbeddingConfig,
        description="Embedding provider configuration",
    )

    # Store references
    stores: StoresConfig = Field(default_factory=StoresConfig)

    # Chunking settings
    chunk_size: int = Field(default=512, description="Document chunk size")
    chunk_overlap: int = Field(default=64, description="Chunk overlap")

    # Multi-tenancy
    default_tenant_id: str = Field(default="default")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StackConfig":
        """Create config from dict with environment variable substitution."""
        resolved = replace_env_vars(data)
        return cls.model_validate(resolved)

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        """Generate sample configuration for documentation."""
        return {
            "version": 1,
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
            },
            "storage_backends": {
                "default": PostgresStorageConfig.sample_config(),
            },
            "vector_backends": {
                "vectors": PgVectorConfig.sample_config(),
            },
            "file_backends": {
                "files": LocalFileStorageConfig.sample_config(),
            },
            "cache_backends": {
                "cache": RedisCacheConfig.sample_config(),
            },
            "inference": VllmInferenceConfig.sample_config(),
            "embedding": {
                "type": "vllm",
                "base_url": "${env.VLLM_BASE_URL:=http://localhost:8000}",
                "model": "text-embedding-3-small",
                "dimension": 1536,
            },
        }


# -----------------------------------------------------------------------------
# Settings (for simple environment-based config)
# -----------------------------------------------------------------------------


class Settings(BaseSettings):
    """Simple settings for environment-based configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_ORCHESTRATOR_",
        env_file=".env",
        case_sensitive=False,
    )

    # Config file path (if using YAML config)
    config_file: Path | None = Field(
        default=None,
        description="Path to YAML configuration file",
    )

    # Server settings (used if no config file)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    debug: bool = Field(default=False)

    # Quick-start settings (used to build default StackConfig)
    vllm_base_url: str = Field(default="http://localhost:8000")
    vllm_api_key: str | None = Field(default=None)
    database_url: str = Field(default="sqlite+aiosqlite:///./orchestrator.db")
    file_storage_path: Path = Field(default=Path("./file_storage"))

    def to_stack_config(self) -> StackConfig:
        """Convert simple settings to full StackConfig."""
        # Parse database URL to determine backend type
        if "sqlite" in self.database_url:
            db_path = self.database_url.split("///")[-1]
            storage_backend: StorageBackendConfig = SqliteStorageConfig(db_path=db_path)
            vector_backend: VectorBackendConfig = SqliteVectorConfig(
                db_path=str(self.file_storage_path / "vectors.db")
            )
        else:
            # Assume PostgreSQL
            storage_backend = PostgresStorageConfig()
            vector_backend = PgVectorConfig()

        return StackConfig(
            server=ServerConfig(host=self.host, port=self.port),
            storage_backends={"default": storage_backend},
            vector_backends={"vectors": vector_backend},
            file_backends={
                "files": LocalFileStorageConfig(base_path=self.file_storage_path)
            },
            inference=VllmInferenceConfig(
                base_url=self.vllm_base_url,
                api_key=self.vllm_api_key,
            ),
            embedding=VllmEmbeddingConfig(base_url=self.vllm_base_url),
        )


# Global settings instance
settings = Settings()
