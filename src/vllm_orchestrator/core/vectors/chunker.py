"""Document chunking for vector storage.

Uses tokenization endpoints for model-accurate chunking.
"""

import hashlib
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from vllm_orchestrator.vllm_client import VllmClient


class ChunkMetadata(BaseModel):
    """Backend metadata for a chunk.

    This metadata is for backend functionality and indexing,
    not included in inference context.
    """

    chunk_id: str | None = None
    document_id: str | None = None
    source: str | None = None
    created_timestamp: int | None = None
    updated_timestamp: int | None = None
    chunk_window: str | None = None
    content_token_count: int | None = None
    metadata_token_count: int | None = None


class Chunk(BaseModel):
    """A chunk of content from file processing."""

    content: str
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_metadata: ChunkMetadata

    @property
    def document_id(self) -> str | None:
        """Returns document_id from metadata or chunk_metadata."""
        doc_id = self.metadata.get("document_id")
        if doc_id is not None:
            return str(doc_id)
        if self.chunk_metadata is not None:
            return self.chunk_metadata.document_id
        return None


class EmbeddedChunk(Chunk):
    """A chunk with its embedding vector."""

    embedding: list[float]
    embedding_model: str
    embedding_dimension: int


class ChunkingStrategyStaticConfig(BaseModel):
    """Configuration for static chunking strategy."""

    chunk_overlap_tokens: int = 400
    max_chunk_size_tokens: int = Field(800, ge=100, le=4096)


class ChunkingStrategyStatic(BaseModel):
    """Static chunking strategy with configurable parameters."""

    type: Literal["static"] = "static"
    static: ChunkingStrategyStaticConfig


class ChunkingStrategyAuto(BaseModel):
    """Automatic chunking strategy."""

    type: Literal["auto"] = "auto"


ChunkingStrategy = ChunkingStrategyAuto | ChunkingStrategyStatic


def generate_chunk_id(content: str, document_id: str, window: str) -> str:
    """Generate a unique chunk ID based on content hash."""
    hash_input = f"{document_id}:{window}:{content[:100]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:24]


class TokenChunker:
    """Chunker that uses tokenization endpoints for accurate token counting."""

    def __init__(self, vllm_client: VllmClient, model: str):
        """Initialize chunker with inference client.

        Args:
            vllm_client: Client for inference server
            model: Model name to use for tokenization
        """
        self._vllm_client = vllm_client
        self._model = model

    async def tokenize_text(self, text: str) -> tuple[list[int], int]:
        """Tokenize text using the inference server's tokenization endpoint.

        Args:
            text: Text to tokenize

        Returns:
            Tuple of (token_ids, token_count)
        """
        # Use the inference server's tokenization endpoint
        response = await self._vllm_client._client.post(
            "/tokenize",
            json={
                "model": self._model,
                "prompt": text,
                "add_special_tokens": False,
                "return_token_strs": False,
            },
        )
        response.raise_for_status()
        result = await response.json()
        return result["tokens"], result["count"]

    async def detokenize_tokens(self, tokens: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        response = await self._vllm_client._client.post(
            "/detokenize",
            json={
                "model": self._model,
                "tokens": tokens,
            },
        )
        response.raise_for_status()
        result = await response.json()
        return result["prompt"]

    async def make_overlapped_chunks(
        self,
        document_id: str,
        text: str,
        window_len: int,
        overlap_len: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Create overlapping chunks using the model's tokenizer.

        Args:
            document_id: ID of the source document
            text: Text content to chunk
            window_len: Chunk size in tokens
            overlap_len: Overlap between chunks in tokens
            metadata: Additional metadata for each chunk

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        metadata = metadata or {}

        # Tokenize the full text using the inference server
        tokens, total_count = await self.tokenize_text(text)

        # Calculate metadata token count
        metadata_token_count = 0
        if metadata:
            try:
                metadata_string = str(metadata)
                _, metadata_token_count = await self.tokenize_text(metadata_string)
            except Exception:
                metadata_token_count = 0

        chunks: list[Chunk] = []
        step_size = max(1, window_len - overlap_len)
        current_time = int(time.time())

        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i : i + window_len]

            # Convert tokens back to text
            chunk_text = await self.detokenize_tokens(chunk_tokens)
            chunk_window = f"{i}-{i + len(chunk_tokens)}"
            chunk_id = generate_chunk_id(chunk_text, document_id, chunk_window)

            chunk_metadata_dict = metadata.copy()
            chunk_metadata_dict["chunk_id"] = chunk_id
            chunk_metadata_dict["document_id"] = document_id
            chunk_metadata_dict["token_count"] = len(chunk_tokens)

            backend_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                source=metadata.get("source"),
                created_timestamp=current_time,
                updated_timestamp=current_time,
                chunk_window=chunk_window,
                content_token_count=len(chunk_tokens),
                metadata_token_count=metadata_token_count,
            )

            chunks.append(
                Chunk(
                    content=chunk_text.strip(),
                    chunk_id=chunk_id,
                    metadata=chunk_metadata_dict,
                    chunk_metadata=backend_metadata,
                )
            )

        return chunks

    async def chunk_document(
        self,
        document_id: str,
        text: str,
        strategy: ChunkingStrategy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Chunk a document using the specified strategy.

        Args:
            document_id: ID of the source document
            text: Text content to chunk
            strategy: Chunking strategy (auto or static)
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        if strategy is None:
            strategy = ChunkingStrategyAuto()

        if isinstance(strategy, ChunkingStrategyAuto):
            # Auto strategy uses sensible defaults
            window_len = 800
            overlap_len = 400
        else:
            window_len = strategy.static.max_chunk_size_tokens
            overlap_len = strategy.static.chunk_overlap_tokens

        return await self.make_overlapped_chunks(
            document_id=document_id,
            text=text,
            window_len=window_len,
            overlap_len=overlap_len,
            metadata=metadata,
        )


# Fallback chunker for when inference server is unavailable
def simple_char_chunks(
    document_id: str,
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Fallback character-based chunking.

    Used when tokenization endpoints are unavailable.

    Args:
        document_id: ID of the source document
        text: Text to chunk
        chunk_size: Chunk size in characters
        overlap: Overlap in characters
        metadata: Additional metadata

    Returns:
        List of Chunk objects
    """
    if not text:
        return []

    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_time = int(time.time())
    step_size = max(1, chunk_size - overlap)

    for i in range(0, len(text), step_size):
        chunk_text = text[i : i + chunk_size].strip()
        if not chunk_text:
            continue

        chunk_window = f"{i}-{i + len(chunk_text)}"
        chunk_id = generate_chunk_id(chunk_text, document_id, chunk_window)

        chunk_metadata_dict = metadata.copy()
        chunk_metadata_dict["chunk_id"] = chunk_id
        chunk_metadata_dict["document_id"] = document_id
        chunk_metadata_dict["char_count"] = len(chunk_text)

        # Rough token estimate (4 chars per token)
        estimated_tokens = len(chunk_text) // 4

        backend_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            source=metadata.get("source"),
            created_timestamp=current_time,
            updated_timestamp=current_time,
            chunk_window=chunk_window,
            content_token_count=estimated_tokens,
        )

        chunks.append(
            Chunk(
                content=chunk_text,
                chunk_id=chunk_id,
                metadata=chunk_metadata_dict,
                chunk_metadata=backend_metadata,
            )
        )

    return chunks
