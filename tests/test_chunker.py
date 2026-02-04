"""Tests for TokenChunker."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_orchestrator.core.vectors.chunker import (
    ChunkingStrategyAuto,
    ChunkingStrategyStatic,
    ChunkingStrategyStaticConfig,
    TokenChunker,
    simple_char_chunks,
)
from vllm_orchestrator.vllm_client import VllmClient


@pytest.fixture
def mock_vllm_client():
    """Create a mock vLLM client."""
    client = MagicMock(spec=VllmClient)

    # Mock HTTP client
    mock_http_client = AsyncMock()
    client._client = mock_http_client

    return client


@pytest.fixture
def mock_tokenize_response():
    """Mock tokenization response."""
    # Mock response for "Hello world, this is a test document"
    return {
        "tokens": [15043, 1917, 11, 420, 374, 264, 1296, 2246],  # Example token IDs
        "count": 8,
    }


@pytest.fixture
def mock_detokenize_response():
    """Mock detokenization response."""
    return {"prompt": "Hello world, this"}


@pytest.mark.asyncio
async def test_token_chunker_initialization(mock_vllm_client):
    """Test TokenChunker initialization."""
    chunker = TokenChunker(mock_vllm_client, "test-model")

    assert chunker._vllm_client == mock_vllm_client
    assert chunker._model == "test-model"


@pytest.mark.asyncio
async def test_tokenize_text(mock_vllm_client, mock_tokenize_response):
    """Test text tokenization."""
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = mock_tokenize_response
    mock_vllm_client._client.post.return_value = mock_response

    chunker = TokenChunker(mock_vllm_client, "test-model")
    tokens, count = await chunker.tokenize_text("Hello world, this is a test")

    # Verify request
    mock_vllm_client._client.post.assert_called_once_with(
        "/tokenize",
        json={
            "model": "test-model",
            "prompt": "Hello world, this is a test",
            "add_special_tokens": False,
            "return_token_strs": False,
        },
    )

    # Verify response
    assert tokens == mock_tokenize_response["tokens"]
    assert count == mock_tokenize_response["count"]


@pytest.mark.asyncio
async def test_detokenize_tokens(mock_vllm_client, mock_detokenize_response):
    """Test token detokenization."""
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = mock_detokenize_response
    mock_vllm_client._client.post.return_value = mock_response

    chunker = TokenChunker(mock_vllm_client, "test-model")
    text = await chunker.detokenize_tokens([15043, 1917, 11, 420])

    # Verify request
    mock_vllm_client._client.post.assert_called_once_with(
        "/detokenize",
        json={
            "model": "test-model",
            "tokens": [15043, 1917, 11, 420],
        },
    )

    # Verify response
    assert text == mock_detokenize_response["prompt"]


@pytest.mark.asyncio
async def test_chunk_document_empty_text(mock_vllm_client):
    """Test chunking empty text."""
    chunker = TokenChunker(mock_vllm_client, "test-model")
    chunks = await chunker.chunk_document("doc_123", "")

    assert chunks == []
    # Should not make any API calls for empty text
    mock_vllm_client._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_document_auto_strategy(mock_vllm_client):
    """Test document chunking with auto strategy."""
    # Setup mock responses
    tokenize_response = AsyncMock()
    tokenize_response.raise_for_status = MagicMock()
    tokenize_response.json = AsyncMock(
        return_value={
            "tokens": list(range(1000)),  # 1000 tokens
            "count": 1000,
        }
    )

    detokenize_response = AsyncMock()
    detokenize_response.raise_for_status = MagicMock()
    detokenize_response.json = AsyncMock(return_value={"prompt": "chunk text"})

    # Use a callable side effect that always returns the appropriate response
    def mock_post(url, **kwargs):
        if url == "/tokenize":
            return tokenize_response
        elif url == "/detokenize":
            return detokenize_response
        else:
            raise ValueError(f"Unexpected URL: {url}")

    mock_vllm_client._client.post.side_effect = mock_post

    chunker = TokenChunker(mock_vllm_client, "test-model")
    chunks = await chunker.chunk_document(
        "doc_123",
        "A long document that will be split into chunks",
        strategy=ChunkingStrategyAuto(),
        metadata={"source": "test"},
    )

    # With 1000 tokens, window_len=800, overlap=400, we should get 3 chunks
    # Chunk 1: 0-800, Chunk 2: 400-1000, Chunk 3: 800-1000
    assert len(chunks) == 3

    # Check chunk properties
    for chunk in chunks:
        assert chunk.document_id == "doc_123"
        assert chunk.content == "chunk text"
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["document_id"] == "doc_123"
        assert chunk.chunk_metadata is not None
        assert chunk.chunk_metadata.document_id == "doc_123"


@pytest.mark.asyncio
async def test_chunk_document_static_strategy(mock_vllm_client):
    """Test document chunking with static strategy."""
    # Setup mock responses for smaller chunks
    tokenize_response = AsyncMock()
    tokenize_response.raise_for_status = MagicMock()
    tokenize_response.json = AsyncMock(
        return_value={
            "tokens": list(range(500)),  # 500 tokens
            "count": 500,
        }
    )

    detokenize_response = AsyncMock()
    detokenize_response.raise_for_status = MagicMock()
    detokenize_response.json = AsyncMock(return_value={"prompt": "small chunk"})

    # Use a callable side effect that always returns the appropriate response
    def mock_post(url, **kwargs):
        if url == "/tokenize":
            return tokenize_response
        elif url == "/detokenize":
            return detokenize_response
        else:
            raise ValueError(f"Unexpected URL: {url}")

    mock_vllm_client._client.post.side_effect = mock_post

    chunker = TokenChunker(mock_vllm_client, "test-model")
    strategy = ChunkingStrategyStatic(
        static=ChunkingStrategyStaticConfig(
            max_chunk_size_tokens=200, chunk_overlap_tokens=50
        )
    )

    chunks = await chunker.chunk_document(
        "doc_456", "A document to be split into smaller chunks", strategy=strategy
    )

    # With 500 tokens, window_len=200, overlap=50, step=150
    # We should get chunks at: 0-200, 150-350, 300-500 = 3 chunks
    assert len(chunks) >= 2  # At least 2 chunks


def test_simple_char_chunks():
    """Test fallback character-based chunking."""
    text = "This is a test document that should be split into multiple chunks for testing purposes."

    chunks = simple_char_chunks(
        "doc_789", text, chunk_size=20, overlap=5, metadata={"type": "fallback"}
    )

    assert len(chunks) > 1  # Should create multiple chunks

    for chunk in chunks:
        assert chunk.document_id == "doc_789"
        assert len(chunk.content) <= 20  # Respects chunk size
        assert chunk.metadata["type"] == "fallback"
        assert chunk.chunk_metadata is not None
        assert chunk.chunk_metadata.content_token_count is not None


def test_simple_char_chunks_empty():
    """Test character chunking with empty text."""
    chunks = simple_char_chunks("doc_empty", "")
    assert chunks == []


def test_simple_char_chunks_short_text():
    """Test character chunking with text shorter than chunk size."""
    chunks = simple_char_chunks("doc_short", "Short text", chunk_size=100)

    assert len(chunks) == 1
    assert chunks[0].content == "Short text"
    assert chunks[0].document_id == "doc_short"
