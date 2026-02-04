"""Object storage backends for file persistence.

Supports multiple backends:
- Local filesystem / PVC (Kubernetes persistent volumes)
- S3-compatible (AWS S3, MinIO, Ceph, etc.)
- GCS (Google Cloud Storage)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO

import aiofiles
import aiofiles.os
from pydantic import BaseModel, Field

from vllm_orchestrator.config import (
    FileStorageBackendConfig,
    LocalFileStorageConfig,
    S3FileStorageConfig,
)


class ObjectMetadata(BaseModel):
    """Metadata for a stored object."""

    key: str
    size: int
    content_type: str | None = None
    etag: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ObjectStorage(ABC):
    """Abstract base class for object storage backends.

    Provides a unified interface for storing and retrieving binary objects
    across different storage backends (local, S3, GCS, etc.).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create buckets, directories, etc.)."""
        ...

    @abstractmethod
    async def put(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ObjectMetadata:
        """Store an object.

        Args:
            key: Object key (path-like identifier)
            content: Object content as bytes
            content_type: MIME type
            metadata: Custom metadata

        Returns:
            ObjectMetadata for the stored object
        """
        ...

    @abstractmethod
    async def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ObjectMetadata:
        """Store an object from a stream.

        Args:
            key: Object key
            stream: File-like object to read from
            content_type: MIME type
            metadata: Custom metadata

        Returns:
            ObjectMetadata for the stored object
        """
        ...

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve object content.

        Args:
            key: Object key

        Returns:
            Object content as bytes

        Raises:
            FileNotFoundError: If object doesn't exist
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an object.

        Args:
            key: Object key

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an object exists.

        Args:
            key: Object key

        Returns:
            True if exists
        """
        ...

    @abstractmethod
    async def get_metadata(self, key: str) -> ObjectMetadata | None:
        """Get object metadata without fetching content.

        Args:
            key: Object key

        Returns:
            ObjectMetadata if exists, None otherwise
        """
        ...

    async def close(self) -> None:
        """Close any open connections."""
        pass


class LocalObjectStorage(ObjectStorage):
    """Local filesystem / PVC object storage.

    Works for both local development and Kubernetes PersistentVolumeClaims.
    In Kubernetes, just mount the PVC to a path and use that as base_path.
    """

    def __init__(
        self,
        base_path: Path | str,
        max_file_size_bytes: int = 512 * 1024 * 1024,
    ):
        self._base_path = Path(base_path)
        self._max_file_size = max_file_size_bytes

    async def initialize(self) -> None:
        """Create base directory if it doesn't exist."""
        await aiofiles.os.makedirs(self._base_path, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert object key to filesystem path."""
        # Normalize key to prevent directory traversal
        normalized = Path(key).as_posix().lstrip("/")
        return self._base_path / normalized

    async def put(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ObjectMetadata:
        if len(content) > self._max_file_size:
            raise ValueError(
                f"Content size ({len(content)}) exceeds maximum ({self._max_file_size})"
            )

        file_path = self._key_to_path(key)
        await aiofiles.os.makedirs(file_path.parent, exist_ok=True)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        # Store metadata in sidecar file if provided
        if metadata or content_type:
            await self._write_metadata(file_path, content_type, metadata)

        return ObjectMetadata(
            key=key,
            size=len(content),
            content_type=content_type,
            metadata=metadata or {},
        )

    async def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ObjectMetadata:
        file_path = self._key_to_path(key)
        await aiofiles.os.makedirs(file_path.parent, exist_ok=True)

        total_size = 0
        chunk_size = 64 * 1024  # 64KB

        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                if total_size + len(chunk) > self._max_file_size:
                    await aiofiles.os.remove(file_path)
                    raise ValueError(
                        f"File exceeds maximum size ({self._max_file_size})"
                    )
                await f.write(chunk)
                total_size += len(chunk)

        if metadata or content_type:
            await self._write_metadata(file_path, content_type, metadata)

        return ObjectMetadata(
            key=key,
            size=total_size,
            content_type=content_type,
            metadata=metadata or {},
        )

    async def get(self, key: str) -> bytes:
        file_path = self._key_to_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Object not found: {key}")

        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def delete(self, key: str) -> bool:
        file_path = self._key_to_path(key)
        if not file_path.exists():
            return False

        await aiofiles.os.remove(file_path)

        # Remove metadata sidecar if exists
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        if meta_path.exists():
            await aiofiles.os.remove(meta_path)

        # Clean up empty parent directories
        await self._cleanup_empty_dirs(file_path.parent)

        return True

    async def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    async def get_metadata(self, key: str) -> ObjectMetadata | None:
        file_path = self._key_to_path(key)
        if not file_path.exists():
            return None

        stat = await aiofiles.os.stat(file_path)
        content_type, metadata = await self._read_metadata(file_path)

        return ObjectMetadata(
            key=key,
            size=stat.st_size,
            content_type=content_type,
            metadata=metadata,
        )

    async def _write_metadata(
        self,
        file_path: Path,
        content_type: str | None,
        metadata: dict[str, str] | None,
    ) -> None:
        """Write metadata to sidecar file."""
        import json

        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        meta = {"content_type": content_type, "metadata": metadata or {}}
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(meta))

    async def _read_metadata(
        self, file_path: Path
    ) -> tuple[str | None, dict[str, str]]:
        """Read metadata from sidecar file."""
        import json

        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        if not meta_path.exists():
            return None, {}
        async with aiofiles.open(meta_path) as f:
            meta = json.loads(await f.read())
        return meta.get("content_type"), meta.get("metadata", {})

    async def _cleanup_empty_dirs(self, path: Path) -> None:
        """Remove empty directories up to base_path."""
        while path != self._base_path:
            try:
                await aiofiles.os.rmdir(path)
                path = path.parent
            except OSError:
                break


class S3ObjectStorage(ObjectStorage):
    """S3-compatible object storage.

    Works with AWS S3, MinIO, Ceph, DigitalOcean Spaces, etc.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        max_file_size_bytes: int = 512 * 1024 * 1024,
    ):
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._region = region
        self._endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._max_file_size = max_file_size_bytes
        self._client: Any = None

    def _full_key(self, key: str) -> str:
        """Get full S3 key with prefix."""
        if self._prefix:
            return f"{self._prefix}/{key.lstrip('/')}"
        return key.lstrip("/")

    async def initialize(self) -> None:
        """Initialize S3 client."""
        try:
            import aioboto3
        except ImportError:
            raise ImportError(
                "aioboto3 is required for S3 storage. "
                "Install with: pip install aioboto3"
            )

        session = aioboto3.Session()
        config = {}
        if self._region:
            config["region_name"] = self._region
        if self._endpoint_url:
            config["endpoint_url"] = self._endpoint_url
        if self._access_key_id:
            config["aws_access_key_id"] = self._access_key_id
        if self._secret_access_key:
            config["aws_secret_access_key"] = self._secret_access_key

        self._session = session
        self._config = config

    async def _get_client(self):
        """Get S3 client context manager."""

        return self._session.client("s3", **self._config)

    async def put(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ObjectMetadata:
        if len(content) > self._max_file_size:
            raise ValueError(f"Content exceeds maximum size ({self._max_file_size})")

        full_key = self._full_key(key)
        extra_args: dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        async with await self._get_client() as client:
            response = await client.put_object(
                Bucket=self._bucket,
                Key=full_key,
                Body=content,
                **extra_args,
            )

        return ObjectMetadata(
            key=key,
            size=len(content),
            content_type=content_type,
            etag=response.get("ETag", "").strip('"'),
            metadata=metadata or {},
        )

    async def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ObjectMetadata:
        # Read stream into memory for simplicity
        # For large files, use multipart upload
        content = stream.read()
        return await self.put(
            key, content, content_type=content_type, metadata=metadata
        )

    async def get(self, key: str) -> bytes:
        full_key = self._full_key(key)
        async with await self._get_client() as client:
            try:
                response = await client.get_object(Bucket=self._bucket, Key=full_key)
                async with response["Body"] as stream:
                    return await stream.read()
            except client.exceptions.NoSuchKey:
                raise FileNotFoundError(f"Object not found: {key}")

    async def delete(self, key: str) -> bool:
        full_key = self._full_key(key)
        async with await self._get_client() as client:
            try:
                await client.delete_object(Bucket=self._bucket, Key=full_key)
                return True
            except Exception:
                return False

    async def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        async with await self._get_client() as client:
            try:
                await client.head_object(Bucket=self._bucket, Key=full_key)
                return True
            except Exception:
                return False

    async def get_metadata(self, key: str) -> ObjectMetadata | None:
        full_key = self._full_key(key)
        async with await self._get_client() as client:
            try:
                response = await client.head_object(Bucket=self._bucket, Key=full_key)
                return ObjectMetadata(
                    key=key,
                    size=response["ContentLength"],
                    content_type=response.get("ContentType"),
                    etag=response.get("ETag", "").strip('"'),
                    metadata=response.get("Metadata", {}),
                )
            except Exception:
                return None


class GCSObjectStorage(ObjectStorage):
    """Google Cloud Storage backend.

    Placeholder - implement with google-cloud-storage when needed.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        project: str | None = None,
        credentials_path: str | None = None,
        max_file_size_bytes: int = 512 * 1024 * 1024,
    ):
        raise NotImplementedError(
            "GCS storage not yet implemented. "
            "Use S3-compatible storage with GCS interoperability, or contribute an implementation!"
        )

    async def initialize(self) -> None:
        raise NotImplementedError

    async def put(self, key: str, content: bytes, **kwargs) -> ObjectMetadata:
        raise NotImplementedError

    async def put_stream(self, key: str, stream: BinaryIO, **kwargs) -> ObjectMetadata:
        raise NotImplementedError

    async def get(self, key: str) -> bytes:
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        raise NotImplementedError

    async def get_metadata(self, key: str) -> ObjectMetadata | None:
        raise NotImplementedError


def create_object_storage(config: FileStorageBackendConfig) -> ObjectStorage:
    """Factory function to create object storage from config."""
    match config:
        case LocalFileStorageConfig():
            return LocalObjectStorage(
                base_path=config.base_path,
                max_file_size_bytes=config.max_file_size_mb * 1024 * 1024,
            )
        case S3FileStorageConfig():
            return S3ObjectStorage(
                bucket=config.bucket,
                prefix=config.prefix,
                region=config.region,
                endpoint_url=config.endpoint_url,
                access_key_id=config.access_key_id,
                secret_access_key=config.secret_access_key,
                max_file_size_bytes=config.max_file_size_mb * 1024 * 1024,
            )
        case _:
            raise ValueError(f"Unknown storage backend type: {type(config)}")
