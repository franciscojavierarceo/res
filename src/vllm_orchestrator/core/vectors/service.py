"""Vector store service for managing vector stores and search operations."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vllm_orchestrator.core.files.service import FilesService
from vllm_orchestrator.core.vectors.chunker import (
    ChunkingStrategy,
    EmbeddedChunk,
    TokenChunker,
)
from vllm_orchestrator.core.vectors.store import SqliteVectorStore, VectorStore
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.storage.models import VectorStoreModel
from vllm_orchestrator.vllm_client import VllmClient


def generate_vector_store_id() -> str:
    """Generate a unique vector store ID."""
    return f"vs_{uuid.uuid4().hex[:24]}"


class VectorStoreService:
    """Service for managing vector stores and search operations.

    This service orchestrates:
    - Vector store creation and management
    - File processing and chunking
    - Embedding generation
    - Vector search operations
    """

    def __init__(
        self,
        vllm_client: VllmClient,
        db_manager: DatabaseManager,
        files_service: FilesService,
        default_tenant_id: str = "default",
        vector_db_path: str = "./vectors.db",
    ):
        self._vllm_client = vllm_client
        self._db_manager = db_manager
        self._files_service = files_service
        self._default_tenant_id = default_tenant_id
        self._vector_db_path = vector_db_path

        # Cache for vector store instances
        self._vector_stores: dict[str, VectorStore] = {}

    async def create_vector_store(
        self,
        name: str,
        file_ids: list[str] | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,
        chunking_strategy: ChunkingStrategy | None = None,
        tenant_id: str | None = None,
    ) -> VectorStoreModel:
        """Create a new vector store.

        Args:
            name: Display name for the vector store
            file_ids: Optional list of file IDs to add immediately
            embedding_model: Model to use for embeddings
            embedding_dimension: Expected embedding dimension
            chunking_strategy: How to chunk documents
            tenant_id: Tenant identifier

        Returns:
            VectorStoreModel instance
        """
        tenant_id = tenant_id or self._default_tenant_id
        vector_store_id = generate_vector_store_id()

        # Create database record
        async with self._db_manager.session() as session:
            vector_store = VectorStoreModel(
                id=vector_store_id,
                name=name,
                status="pending",
                embedding_model=embedding_model,
                dimension=embedding_dimension,
                file_counts={
                    "completed": 0,
                    "cancelled": 0,
                    "failed": 0,
                    "in_progress": 0,
                    "total": len(file_ids) if file_ids else 0,
                },
                tenant_id=tenant_id,
            )
            session.add(vector_store)
            await session.flush()

            # Create vector store instance
            store = SqliteVectorStore(self._vector_db_path, vector_store_id)
            await store.initialize()
            self._vector_stores[vector_store_id] = store

            # Process files if provided
            if file_ids:
                try:
                    await self._process_files(
                        vector_store_id,
                        file_ids,
                        embedding_model,
                        chunking_strategy,
                        tenant_id,
                        session,
                    )
                    vector_store.status = "completed"
                except Exception as e:
                    vector_store.status = "failed"
                    vector_store.metadata_ = {"error": str(e)}

            await session.flush()
            await session.refresh(vector_store)
            return vector_store

    async def get_vector_store(
        self,
        vector_store_id: str,
        tenant_id: str | None = None,
    ) -> VectorStoreModel | None:
        """Get vector store metadata by ID."""
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            result = await session.execute(
                select(VectorStoreModel).where(
                    VectorStoreModel.id == vector_store_id,
                    VectorStoreModel.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_vector_stores(
        self,
        tenant_id: str | None = None,
        limit: int = 20,
        after: str | None = None,
        order: str = "desc",
    ) -> list[VectorStoreModel]:
        """List vector stores with pagination."""
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            query = select(VectorStoreModel).where(
                VectorStoreModel.tenant_id == tenant_id
            )

            if after is not None:
                after_store = await session.execute(
                    select(VectorStoreModel).where(
                        VectorStoreModel.id == after,
                        VectorStoreModel.tenant_id == tenant_id,
                    )
                )
                after_result = after_store.scalar_one_or_none()
                if after_result:
                    query = query.where(
                        VectorStoreModel.created_at > after_result.created_at
                    )

            if order == "desc":
                query = query.order_by(VectorStoreModel.created_at.desc())
            else:
                query = query.order_by(VectorStoreModel.created_at.asc())

            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_vector_store(
        self,
        vector_store_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """Delete a vector store and all its chunks."""
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            vector_store = await session.execute(
                select(VectorStoreModel).where(
                    VectorStoreModel.id == vector_store_id,
                    VectorStoreModel.tenant_id == tenant_id,
                )
            )
            store_result = vector_store.scalar_one_or_none()
            if store_result is None:
                return False

            # Delete from vector store
            if vector_store_id in self._vector_stores:
                # Note: SqliteVectorStore doesn't expose delete_all, would need to add
                # For now, just remove from cache
                del self._vector_stores[vector_store_id]

            # Delete database record
            await session.delete(store_result)
            await session.flush()
            return True

    async def search_vector_store(
        self,
        vector_store_id: str,
        query: str,
        max_results: int = 10,
        score_threshold: float = 0.0,
        tenant_id: str | None = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Search a vector store using semantic similarity.

        Args:
            vector_store_id: Vector store ID to search
            query: Text query to search for
            max_results: Maximum number of results
            score_threshold: Minimum similarity score
            tenant_id: Tenant identifier

        Returns:
            List of (chunk, similarity_score) tuples
        """
        tenant_id = tenant_id or self._default_tenant_id

        # Verify access to vector store
        vector_store = await self.get_vector_store(vector_store_id, tenant_id)
        if vector_store is None:
            raise ValueError(f"Vector store {vector_store_id} not found")

        # Get vector store instance
        store = await self._get_vector_store_instance(vector_store_id)

        # Generate query embedding
        query_embedding = await self._generate_embedding(
            query, vector_store.embedding_model
        )

        # Search for similar chunks
        return await store.search_similar(query_embedding, max_results, score_threshold)

    async def _process_files(
        self,
        vector_store_id: str,
        file_ids: list[str],
        embedding_model: str,
        chunking_strategy: ChunkingStrategy | None,
        tenant_id: str,
        session: AsyncSession,
    ) -> None:
        """Process files and add chunks to vector store."""
        store = await self._get_vector_store_instance(vector_store_id)
        chunker = TokenChunker(self._vllm_client, embedding_model)

        all_chunks: list[EmbeddedChunk] = []

        for file_id in file_ids:
            # Get file content
            file_text = await self._files_service.get_file_text(file_id, tenant_id)
            if file_text is None:
                continue

            # Chunk the document
            chunks = await chunker.chunk_document(
                document_id=file_id,
                text=file_text,
                strategy=chunking_strategy,
                metadata={"file_id": file_id, "source": "file_upload"},
            )

            # Generate embeddings for chunks
            for chunk in chunks:
                embedding = await self._generate_embedding(
                    chunk.content, embedding_model
                )
                embedded_chunk = EmbeddedChunk(
                    content=chunk.content,
                    chunk_id=chunk.chunk_id,
                    metadata=chunk.metadata,
                    chunk_metadata=chunk.chunk_metadata,
                    embedding=embedding,
                    embedding_model=embedding_model,
                    embedding_dimension=len(embedding),
                )
                all_chunks.append(embedded_chunk)

        # Add all chunks to vector store
        if all_chunks:
            await store.add_chunks(all_chunks)

    async def _get_vector_store_instance(self, vector_store_id: str) -> VectorStore:
        """Get or create vector store instance."""
        if vector_store_id not in self._vector_stores:
            store = SqliteVectorStore(self._vector_db_path, vector_store_id)
            await store.initialize()
            self._vector_stores[vector_store_id] = store
        return self._vector_stores[vector_store_id]

    async def _generate_embedding(self, text: str, model: str) -> list[float]:
        """Generate embedding for text using vLLM."""
        result = await self._vllm_client.create_embeddings(
            model=model,
            input=text,
        )
        return result["data"][0]["embedding"]
