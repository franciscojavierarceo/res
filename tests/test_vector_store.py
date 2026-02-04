"""Tests for vector store functionality."""

import os
import tempfile

import pytest

from vllm_orchestrator.core.vectors.chunker import ChunkMetadata, EmbeddedChunk
from vllm_orchestrator.core.vectors.store import SqliteVectorStore


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
def sample_chunks():
    """Create sample embedded chunks for testing."""
    chunks = []
    for i in range(3):
        chunk = EmbeddedChunk(
            content=f"This is test chunk {i} with some content for testing",
            chunk_id=f"chunk_{i}",
            metadata={
                "document_id": "test_doc",
                "file_id": f"file_{i}",
                "source": "test",
            },
            chunk_metadata=ChunkMetadata(
                chunk_id=f"chunk_{i}",
                document_id="test_doc",
                source="test",
                content_token_count=10 + i,
            ),
            embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],  # Simple test embeddings
            embedding_model="test-model",
            embedding_dimension=4,
        )
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_sqlite_vector_store_initialization(temp_db_path):
    """Test vector store initialization."""
    store = SqliteVectorStore(temp_db_path, "test_store")
    await store.initialize()

    # Verify that tables were created
    import sqlite3

    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE 'vs_test_store_%'
    """)
    tables = cursor.fetchall()
    assert len(tables) > 0  # Should have at least the metadata table
    conn.close()


@pytest.mark.asyncio
async def test_add_and_get_chunks_fallback_mode(temp_db_path, sample_chunks):
    """Test adding and retrieving chunks using JSON fallback mode."""
    store = SqliteVectorStore(temp_db_path, "test_store")

    # Initialize store (will naturally use fallback mode if sqlite-vec not available)
    await store.initialize()

    # Add chunks
    await store.add_chunks(sample_chunks)

    # Get individual chunks
    for chunk in sample_chunks:
        retrieved = await store.get_chunk(chunk.chunk_id)
        assert retrieved is not None
        assert retrieved.chunk_id == chunk.chunk_id
        assert retrieved.content == chunk.content
        assert retrieved.embedding == chunk.embedding


@pytest.mark.asyncio
async def test_search_similar_fallback_mode(temp_db_path, sample_chunks):
    """Test similarity search using JSON fallback mode."""
    store = SqliteVectorStore(temp_db_path, "test_store")

    # Initialize store
    await store.initialize()

    # Add chunks
    await store.add_chunks(sample_chunks)

    # Search for similar chunks
    # Use an embedding similar to the first chunk
    query_embedding = [
        0.05,
        0.1,
        0.15,
        0.2,
    ]  # Close to first chunk: [0.0, 0.0, 0.0, 0.0]

    results = await store.search_similar(query_embedding, limit=2)

    # Should return results
    assert len(results) <= 2
    assert len(results) > 0

    # Results should be tuples of (chunk, similarity_score)
    chunk, score = results[0]
    assert isinstance(chunk, EmbeddedChunk)
    assert isinstance(score, float)
    assert score >= 0.0  # Similarity scores should be non-negative


@pytest.mark.asyncio
async def test_delete_chunks(temp_db_path, sample_chunks):
    """Test deleting chunks."""
    store = SqliteVectorStore(temp_db_path, "test_store")

    # Initialize store
    await store.initialize()

    # Add chunks
    await store.add_chunks(sample_chunks)

    # Delete some chunks
    chunk_ids_to_delete = [sample_chunks[0].chunk_id, sample_chunks[1].chunk_id]
    deleted_count = await store.delete_chunks(chunk_ids_to_delete)

    assert deleted_count == 2

    # Verify chunks were deleted
    assert await store.get_chunk(sample_chunks[0].chunk_id) is None
    assert await store.get_chunk(sample_chunks[1].chunk_id) is None
    assert await store.get_chunk(sample_chunks[2].chunk_id) is not None


@pytest.mark.asyncio
async def test_search_with_score_threshold(temp_db_path, sample_chunks):
    """Test similarity search with score threshold."""
    store = SqliteVectorStore(temp_db_path, "test_store")

    # Initialize store
    await store.initialize()

    # Add chunks
    await store.add_chunks(sample_chunks)

    # Search with high score threshold (should return fewer/no results)
    query_embedding = [0.05, 0.1, 0.15, 0.2]
    results_high_threshold = await store.search_similar(
        query_embedding, limit=10, score_threshold=0.9
    )

    # Search with low score threshold (should return more results)
    results_low_threshold = await store.search_similar(
        query_embedding, limit=10, score_threshold=0.0
    )

    # Lower threshold should return same or more results
    assert len(results_low_threshold) >= len(results_high_threshold)


@pytest.mark.asyncio
async def test_empty_vector_store(temp_db_path):
    """Test operations on empty vector store."""
    store = SqliteVectorStore(temp_db_path, "empty_store")

    # Initialize store
    await store.initialize()

    # Search empty store
    results = await store.search_similar([0.1, 0.2, 0.3, 0.4])
    assert results == []

    # Get non-existent chunk
    chunk = await store.get_chunk("nonexistent")
    assert chunk is None

    # Delete from empty store
    deleted = await store.delete_chunks(["nonexistent"])
    assert deleted == 0


@pytest.mark.asyncio
async def test_sqlite_vec_mode_if_available(temp_db_path, sample_chunks):
    """Test sqlite-vec mode if the extension is available."""
    store = SqliteVectorStore(temp_db_path, "test_store")

    # Try to initialize normally (may or may not have sqlite-vec available)
    await store.initialize()

    # Add chunks regardless of mode
    await store.add_chunks(sample_chunks)

    # Basic operations should work in either mode
    retrieved = await store.get_chunk(sample_chunks[0].chunk_id)
    assert retrieved is not None
    assert retrieved.chunk_id == sample_chunks[0].chunk_id

    # Search should work in either mode
    results = await store.search_similar([0.05, 0.1, 0.15, 0.2])
    assert len(results) > 0
