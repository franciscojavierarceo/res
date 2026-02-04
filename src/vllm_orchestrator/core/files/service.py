"""Files service for managing uploaded files."""

import mimetypes
import uuid
from typing import BinaryIO

from sqlalchemy import select

from vllm_orchestrator.core.files.storage import ObjectStorage
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.storage.models import FileModel


def generate_file_id() -> str:
    """Generate a unique file ID."""
    return f"file_{uuid.uuid4().hex[:24]}"


class FilesService:
    """Service for managing file uploads and retrieval.

    Handles:
    - File upload to object storage
    - File metadata persistence to database
    - File retrieval and deletion
    """

    def __init__(
        self,
        object_storage: ObjectStorage,
        db_manager: DatabaseManager,
        default_tenant_id: str = "default",
    ):
        self._storage = object_storage
        self._db_manager = db_manager
        self._default_tenant_id = default_tenant_id

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        purpose: str,
        *,
        tenant_id: str | None = None,
        content_type: str | None = None,
    ) -> FileModel:
        """Upload a file.

        Args:
            content: File content as bytes
            filename: Original filename
            purpose: Purpose of the file (e.g., "assistants", "fine-tune")
            tenant_id: Tenant identifier
            content_type: MIME type (auto-detected if not provided)

        Returns:
            FileModel with file metadata
        """
        tenant_id = tenant_id or self._default_tenant_id
        file_id = generate_file_id()

        # Auto-detect content type if not provided
        if content_type is None:
            content_type, _ = mimetypes.guess_type(filename)

        # Upload to object storage
        storage_key = f"{tenant_id}/{file_id}/{filename}"
        await self._storage.put(
            key=storage_key,
            content=content,
            content_type=content_type,
            metadata={"original_filename": filename, "purpose": purpose},
        )

        # Create database record
        async with self._db_manager.session() as session:
            file_model = FileModel(
                id=file_id,
                filename=filename,
                purpose=purpose,
                bytes=len(content),
                mime_type=content_type,
                storage_backend="object_storage",
                storage_path=storage_key,
                status="uploaded",
                tenant_id=tenant_id,
            )
            session.add(file_model)
            await session.flush()

            # Refresh to get created_at
            await session.refresh(file_model)
            return file_model

    async def upload_file_stream(
        self,
        stream: BinaryIO,
        filename: str,
        purpose: str,
        *,
        tenant_id: str | None = None,
        content_type: str | None = None,
    ) -> FileModel:
        """Upload a file from a stream.

        Args:
            stream: File stream
            filename: Original filename
            purpose: Purpose of the file
            tenant_id: Tenant identifier
            content_type: MIME type

        Returns:
            FileModel with file metadata
        """
        tenant_id = tenant_id or self._default_tenant_id
        file_id = generate_file_id()

        if content_type is None:
            content_type, _ = mimetypes.guess_type(filename)

        storage_key = f"{tenant_id}/{file_id}/{filename}"
        metadata = await self._storage.put_stream(
            key=storage_key,
            stream=stream,
            content_type=content_type,
            metadata={"original_filename": filename, "purpose": purpose},
        )

        async with self._db_manager.session() as session:
            file_model = FileModel(
                id=file_id,
                filename=filename,
                purpose=purpose,
                bytes=metadata.size,
                mime_type=content_type,
                storage_backend="object_storage",
                storage_path=storage_key,
                status="uploaded",
                tenant_id=tenant_id,
            )
            session.add(file_model)
            await session.flush()
            await session.refresh(file_model)
            return file_model

    async def get_file(
        self,
        file_id: str,
        tenant_id: str | None = None,
    ) -> FileModel | None:
        """Get file metadata by ID.

        Args:
            file_id: File ID
            tenant_id: Tenant identifier

        Returns:
            FileModel if found, None otherwise
        """
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            result = await session.execute(
                select(FileModel).where(
                    FileModel.id == file_id,
                    FileModel.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def get_file_content(
        self,
        file_id: str,
        tenant_id: str | None = None,
    ) -> bytes | None:
        """Get file content by ID.

        Args:
            file_id: File ID
            tenant_id: Tenant identifier

        Returns:
            File content as bytes, or None if not found
        """
        file_model = await self.get_file(file_id, tenant_id)
        if file_model is None:
            return None

        try:
            return await self._storage.get(file_model.storage_path)
        except FileNotFoundError:
            return None

    async def list_files(
        self,
        tenant_id: str | None = None,
        *,
        purpose: str | None = None,
        limit: int = 20,
        after: str | None = None,
        order: str = "desc",
    ) -> list[FileModel]:
        """List files with optional filtering.

        Args:
            tenant_id: Tenant identifier
            purpose: Filter by purpose
            limit: Maximum files to return
            after: Pagination cursor
            order: Sort order

        Returns:
            List of FileModel instances
        """
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            query = select(FileModel).where(FileModel.tenant_id == tenant_id)

            if purpose is not None:
                query = query.where(FileModel.purpose == purpose)

            if after is not None:
                after_file = await session.execute(
                    select(FileModel).where(
                        FileModel.id == after,
                        FileModel.tenant_id == tenant_id,
                    )
                )
                after_result = after_file.scalar_one_or_none()
                if after_result:
                    query = query.where(FileModel.created_at > after_result.created_at)

            if order == "desc":
                query = query.order_by(FileModel.created_at.desc())
            else:
                query = query.order_by(FileModel.created_at.asc())

            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_file(
        self,
        file_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """Delete a file.

        Args:
            file_id: File ID to delete
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            file_model = await session.execute(
                select(FileModel).where(
                    FileModel.id == file_id,
                    FileModel.tenant_id == tenant_id,
                )
            )
            file_result = file_model.scalar_one_or_none()
            if file_result is None:
                return False

            # Delete from object storage
            await self._storage.delete(file_result.storage_path)

            # Delete from database
            await session.delete(file_result)
            await session.flush()
            return True

    async def get_file_text(
        self,
        file_id: str,
        tenant_id: str | None = None,
    ) -> str | None:
        """Get file content as text.

        Attempts to decode file content as UTF-8.
        For PDFs and other formats, use specialized extractors.

        Args:
            file_id: File ID
            tenant_id: Tenant identifier

        Returns:
            File content as string, or None if not found/not text
        """
        content = await self.get_file_content(file_id, tenant_id)
        if content is None:
            return None

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # TODO: Add PDF extraction, etc.
            return None
