"""Tests for vector store service."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_orchestrator.core.files.service import FilesService
from vllm_orchestrator.core.vectors.chunker import (
    ChunkingStrategyAuto,
    ChunkingStrategyStatic,
    ChunkingStrategyStaticConfig,
)
from vllm_orchestrator.core.vectors.service import VectorStoreService
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.vllm_client import VllmClient


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
def temp_vector_db_path():
    """Create a temporary vector database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        vector_db_path = f.name
    yield vector_db_path
    # Cleanup
    if os.path.exists(vector_db_path):
        os.unlink(vector_db_path)


@pytest.fixture
async def mock_database_manager(temp_db_path):
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
    async def async_session_context():
        return mock_session

    mock_session_context_manager = MagicMock()
    mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

    db_manager.session.return_value = mock_session_context_manager

    return db_manager, mock_session


@pytest.fixture
def mock_vllm_client():
    """Create a mock vLLM client."""
    client = MagicMock(spec=VllmClient)

    # Mock embedding creation
    async def mock_create_embeddings(model: str, input: str):
        # Return a simple embedding based on text hash for deterministic results
        hash_val = hash(input) % 1000 / 1000.0  # Normalize to [0, 1]
        return {
            "data": [
                {
                    "embedding": [
                        hash_val,
                        hash_val * 0.5,
                        hash_val * 0.25,
                        hash_val * 0.125,
                    ]
                }
            ]
        }

    client.create_embeddings = AsyncMock(side_effect=mock_create_embeddings)
    return client


@pytest.fixture
def mock_files_service():
    """Create a mock files service."""
    files_service = MagicMock(spec=FilesService)

    async def mock_get_file_text(file_id: str, tenant_id: str):
        return f"This is the content of file {file_id} with some text for testing"

    files_service.get_file_text = AsyncMock(side_effect=mock_get_file_text)
    return files_service


@pytest.mark.asyncio
async def test_create_vector_store(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test creating a new vector store."""
    db_manager, mock_session = mock_database_manager

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Mock the vector store model creation
    mock_vector_store = MagicMock()
    mock_vector_store.id = "vs_test123"
    mock_vector_store.name = "test-store"
    mock_vector_store.status = "completed"

    # Create vector store without files
    result = await service.create_vector_store(
        name="test-store",
        embedding_model="text-embedding-test",
        embedding_dimension=1536,
    )

    # Verify database operations were called
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called()
    mock_session.refresh.assert_called()


@pytest.mark.asyncio
async def test_create_vector_store_with_files(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test creating a vector store with initial files."""
    db_manager, mock_session = mock_database_manager

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Create vector store with files
    file_ids = ["file_1", "file_2"]

    result = await service.create_vector_store(
        name="test-store-with-files",
        file_ids=file_ids,
        chunking_strategy=ChunkingStrategyAuto(),
    )

    # Verify files service was called to get file content
    assert mock_files_service.get_file_text.call_count >= 1

    # Note: The exact call counts depend on the internal implementation
    # and may vary as the chunking and embedding logic is refined


@pytest.mark.asyncio
async def test_search_vector_store(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test searching a vector store."""
    db_manager, mock_session = mock_database_manager

    # Mock the vector store lookup
    mock_vector_store = MagicMock()
    mock_vector_store.id = "vs_test123"
    mock_vector_store.embedding_model = "text-embedding-test"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_vector_store
    mock_session.execute.return_value = mock_result

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Mock the internal vector store instance
    mock_store_instance = AsyncMock()
    mock_store_instance.search_similar.return_value = []

    service._vector_stores["vs_test123"] = mock_store_instance

    # Perform search
    results = await service.search_vector_store(
        vector_store_id="vs_test123",
        query="test query",
        max_results=5,
        score_threshold=0.1,
    )

    # Verify embedding generation and search
    mock_vllm_client.create_embeddings.assert_called_once()
    mock_store_instance.search_similar.assert_called_once()
    assert results == []


@pytest.mark.asyncio
async def test_search_nonexistent_vector_store(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test searching a vector store that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result for vector store lookup
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Should raise ValueError for nonexistent vector store
    with pytest.raises(ValueError, match="Vector store .* not found"):
        await service.search_vector_store(
            vector_store_id="nonexistent",
            query="test query",
        )


@pytest.mark.asyncio
async def test_delete_vector_store(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test deleting a vector store."""
    db_manager, mock_session = mock_database_manager

    # Mock the vector store lookup
    mock_vector_store = MagicMock()
    mock_vector_store.id = "vs_test123"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_vector_store
    mock_session.execute.return_value = mock_result

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Add to cache to test cleanup
    service._vector_stores["vs_test123"] = MagicMock()

    # Delete vector store
    result = await service.delete_vector_store("vs_test123")

    assert result is True
    assert "vs_test123" not in service._vector_stores
    mock_session.delete.assert_called_once()
    mock_session.flush.assert_called()


@pytest.mark.asyncio
async def test_delete_nonexistent_vector_store(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test deleting a vector store that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result for vector store lookup
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Should return False for nonexistent vector store
    result = await service.delete_vector_store("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_list_vector_stores(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test listing vector stores."""
    db_manager, mock_session = mock_database_manager

    # Mock vector store list
    mock_stores = [MagicMock(), MagicMock()]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_stores
    mock_session.execute.return_value = mock_result

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # List vector stores
    result = await service.list_vector_stores(limit=10)

    assert result == mock_stores
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_chunking_strategy_static(
    mock_database_manager, mock_vllm_client, mock_files_service, temp_vector_db_path
):
    """Test vector store creation with static chunking strategy."""
    db_manager, mock_session = mock_database_manager

    service = VectorStoreService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        files_service=mock_files_service,
        vector_db_path=temp_vector_db_path,
    )

    # Create with static strategy
    strategy = ChunkingStrategyStatic(
        static=ChunkingStrategyStaticConfig(
            max_chunk_size_tokens=500, chunk_overlap_tokens=100
        )
    )

    result = await service.create_vector_store(
        name="static-chunked-store",
        file_ids=["file_1"],
        chunking_strategy=strategy,
    )

    # Verify it processed the file
    mock_files_service.get_file_text.assert_called_once()
