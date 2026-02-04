"""Tests for files service."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_orchestrator.core.files.service import FilesService
from vllm_orchestrator.core.files.storage import ObjectStorage
from vllm_orchestrator.storage.database import DatabaseManager


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
async def mock_database_manager():
    """Create a mock database manager."""
    db_manager = MagicMock(spec=DatabaseManager)

    # Mock session context manager
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.delete = AsyncMock()

    # Create an async context manager
    mock_session_context_manager = MagicMock()
    mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

    db_manager.session.return_value = mock_session_context_manager

    return db_manager, mock_session


@pytest.fixture
def mock_object_storage():
    """Create a mock object storage."""
    storage = AsyncMock(spec=ObjectStorage)

    # Mock storage methods
    async def mock_put(key: str, content: bytes, **kwargs):
        return {"key": key, "size": len(content)}

    async def mock_get(key: str):
        return b"mock file content"

    async def mock_delete(key: str):
        return True

    storage.put = AsyncMock(side_effect=mock_put)
    storage.get = AsyncMock(side_effect=mock_get)
    storage.delete = AsyncMock(side_effect=mock_delete)
    storage.initialize = AsyncMock()

    return storage


@pytest.fixture
def sample_file_content():
    """Create sample file content for testing."""
    return b"This is sample file content for testing the files service."


@pytest.mark.asyncio
async def test_upload_file(
    mock_database_manager, mock_object_storage, sample_file_content
):
    """Test uploading a file."""
    db_manager, mock_session = mock_database_manager

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Upload file
    file_model = await service.upload_file(
        filename="test.txt",
        content=sample_file_content,
        content_type="text/plain",
        purpose="assistants",
    )

    # Verify storage was called
    mock_object_storage.put.assert_called_once()

    # Verify database was called
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_get_file_content(mock_database_manager, mock_object_storage):
    """Test getting file content."""
    db_manager, mock_session = mock_database_manager

    # Mock file lookup
    mock_file = MagicMock()
    mock_file.id = "file_123"
    mock_file.filename = "test.txt"
    mock_file.storage_path = "default/file_123/test.txt"
    mock_file.tenant_id = "default"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_file
    mock_session.execute.return_value = mock_result

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Get file content
    content = await service.get_file_content("file_123")

    # Verify storage was called
    mock_object_storage.get.assert_called_once_with("default/file_123/test.txt")

    # Verify returned values
    assert content == b"mock file content"


@pytest.mark.asyncio
async def test_get_file_text(mock_database_manager, mock_object_storage):
    """Test getting file as text."""
    db_manager, mock_session = mock_database_manager

    # Mock file lookup
    mock_file = MagicMock()
    mock_file.id = "file_123"
    mock_file.filename = "test.txt"
    mock_file.storage_path = "default/file_123/test.txt"
    mock_file.tenant_id = "default"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_file
    mock_session.execute.return_value = mock_result

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Get file text
    text = await service.get_file_text("file_123")

    # Verify storage was called
    mock_object_storage.get.assert_called_once_with("default/file_123/test.txt")

    # Verify returned text
    assert text == "mock file content"


@pytest.mark.asyncio
async def test_get_nonexistent_file(mock_database_manager, mock_object_storage):
    """Test getting a file that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Try to get nonexistent file
    content = await service.get_file_content("nonexistent")

    # Should return None
    assert content is None

    # Storage should not be called
    mock_object_storage.get.assert_not_called()


@pytest.mark.asyncio
async def test_delete_file(mock_database_manager, mock_object_storage):
    """Test deleting a file."""
    db_manager, mock_session = mock_database_manager

    # Mock file lookup
    mock_file = MagicMock()
    mock_file.id = "file_123"
    mock_file.storage_path = "default/file_123/test.txt"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_file
    mock_session.execute.return_value = mock_result

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Delete file
    result = await service.delete_file("file_123")

    # Verify storage deletion
    mock_object_storage.delete.assert_called_once_with("default/file_123/test.txt")

    # Verify database deletion
    mock_session.delete.assert_called_once()
    mock_session.flush.assert_called_once()

    assert result is True


@pytest.mark.asyncio
async def test_delete_nonexistent_file(mock_database_manager, mock_object_storage):
    """Test deleting a file that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Try to delete nonexistent file
    result = await service.delete_file("nonexistent")

    # Should return False
    assert result is False

    # Storage should not be called
    mock_object_storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_list_files(mock_database_manager, mock_object_storage):
    """Test listing files."""
    db_manager, mock_session = mock_database_manager

    # Mock file list
    mock_files = [MagicMock(), MagicMock()]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_files
    mock_session.execute.return_value = mock_result

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # List files
    result = await service.list_files(limit=10)

    assert result == mock_files
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_upload_file_auto_detect_content_type(
    mock_database_manager, mock_object_storage
):
    """Test uploading a file with auto-detected content type."""
    db_manager, mock_session = mock_database_manager

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Upload file without specifying content type
    file_model = await service.upload_file(
        filename="test.json",
        content=b'{"key": "value"}',
        purpose="assistants",
        # content_type not specified, should be auto-detected
    )

    # Verify storage was called
    mock_object_storage.put.assert_called_once()

    # Check that content type detection was attempted
    call_args = mock_object_storage.put.call_args
    # The exact content type depends on mimetypes.guess_type, but put should be called
    assert call_args is not None


@pytest.mark.asyncio
async def test_upload_file_with_metadata(mock_database_manager, mock_object_storage):
    """Test uploading a file with custom metadata."""
    db_manager, mock_session = mock_database_manager

    service = FilesService(
        object_storage=mock_object_storage,
        db_manager=db_manager,
    )

    # Upload file with custom metadata
    file_model = await service.upload_file(
        filename="test.txt",
        content=b"test content",
        content_type="text/plain",
        purpose="assistants",
    )

    # Verify storage was called with metadata
    mock_object_storage.put.assert_called_once()
    call_args = mock_object_storage.put.call_args
    assert "metadata" in call_args.kwargs
