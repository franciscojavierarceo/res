"""Vector store implementation for file search."""

import asyncio
import json
import struct
from abc import ABC, abstractmethod

import numpy as np

from vllm_orchestrator.core.vectors.chunker import EmbeddedChunk


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    async def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the vector store."""
        ...

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Search for similar chunks by embedding.

        Returns:
            List of (chunk, similarity_score) tuples ordered by similarity desc
        """
        ...

    @abstractmethod
    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID. Returns number deleted."""
        ...

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> EmbeddedChunk | None:
        """Get a chunk by ID."""
        ...


class SqliteVectorStore(VectorStore):
    """Vector store using SQLite with sqlite-vec extension or fallback to JSON."""

    def __init__(self, db_path: str, vector_store_id: str):
        self.db_path = db_path
        self.vector_store_id = vector_store_id
        self.table_prefix = f"vs_{vector_store_id.replace('-', '_')}"
        self.metadata_table = f"{self.table_prefix}_chunks"
        self._sqlite_vec_available = None

    async def initialize(self) -> None:
        """Initialize tables and check for sqlite-vec availability."""
        await asyncio.to_thread(self._init_tables)

    def _init_tables(self) -> None:
        """Initialize SQLite tables in background thread."""
        import sqlite3

        # Try to load sqlite-vec extension
        try:
            import sqlite_vec

            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._sqlite_vec_available = True
            self._init_sqlite_vec_tables(conn)
        except (ImportError, sqlite3.Error):
            # Fallback to regular SQLite
            conn = sqlite3.connect(self.db_path)
            self._sqlite_vec_available = False
            self._init_json_tables(conn)

        conn.close()

    def _init_sqlite_vec_tables(self, conn) -> None:
        """Initialize tables with sqlite-vec extension."""
        cursor = conn.cursor()

        # Metadata table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                id TEXT PRIMARY KEY,
                chunk_data TEXT NOT NULL,
                file_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Vector table (will be created when first vector is added)
        conn.commit()

    def _init_json_tables(self, conn) -> None:
        """Initialize tables for JSON fallback."""
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                id TEXT PRIMARY KEY,
                chunk_data TEXT NOT NULL,
                embedding_json TEXT,
                file_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

    def _serialize_vector(self, vector: list[float]) -> bytes:
        """Serialize vector for sqlite-vec."""
        return struct.pack(f"{len(vector)}f", *vector)

    async def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks using sqlite-vec or JSON fallback."""
        if self._sqlite_vec_available:
            await self._add_chunks_sqlite_vec(chunks)
        else:
            await self._add_chunks_json(chunks)

    async def _add_chunks_sqlite_vec(self, chunks: list[EmbeddedChunk]) -> None:
        """Add chunks using sqlite-vec extension."""

        def _execute_inserts():
            import sqlite3

            import sqlite_vec

            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            cursor = conn.cursor()

            try:
                # Create vector table if needed (first chunk determines dimension)
                if chunks:
                    dimension = len(chunks[0].embedding)
                    vector_table = f"{self.table_prefix}_vectors"
                    cursor.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS {vector_table}
                        USING vec0(embedding FLOAT[{dimension}], chunk_id TEXT)
                    """)

                cursor.execute("BEGIN TRANSACTION")

                for chunk in chunks:
                    # Insert metadata
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.metadata_table}
                        (id, chunk_data, file_id) VALUES (?, ?, ?)
                    """,
                        (
                            chunk.chunk_id,
                            chunk.model_dump_json(),
                            chunk.metadata.get("file_id", "unknown"),
                        ),
                    )

                    # Insert vector
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {vector_table}
                        (chunk_id, embedding) VALUES (?, ?)
                    """,
                        (chunk.chunk_id, self._serialize_vector(chunk.embedding)),
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        await asyncio.to_thread(_execute_inserts)

    async def _add_chunks_json(self, chunks: list[EmbeddedChunk]) -> None:
        """Add chunks using JSON embedding storage."""

        def _execute_inserts():
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute("BEGIN TRANSACTION")

                for chunk in chunks:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.metadata_table}
                        (id, chunk_data, embedding_json, file_id) VALUES (?, ?, ?, ?)
                    """,
                        (
                            chunk.chunk_id,
                            chunk.model_dump_json(),
                            json.dumps(chunk.embedding),
                            chunk.metadata.get("file_id", "unknown"),
                        ),
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        await asyncio.to_thread(_execute_inserts)

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Search for similar vectors."""
        if self._sqlite_vec_available:
            return await self._search_sqlite_vec(
                query_embedding, limit, score_threshold
            )
        else:
            return await self._search_json(query_embedding, limit, score_threshold)

    async def _search_sqlite_vec(
        self, query_embedding: list[float], limit: int, score_threshold: float
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Search using sqlite-vec."""

        def _execute_search():
            import sqlite3

            import sqlite_vec

            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            cursor = conn.cursor()

            try:
                vector_table = f"{self.table_prefix}_vectors"
                query_blob = self._serialize_vector(query_embedding)

                # sqlite-vec similarity search
                cursor.execute(
                    f"""
                    SELECT m.chunk_data, v.distance
                    FROM {vector_table} AS v
                    JOIN {self.metadata_table} AS m ON m.id = v.chunk_id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                """,
                    (query_blob, limit),
                )

                return cursor.fetchall()
            finally:
                conn.close()

        rows = await asyncio.to_thread(_execute_search)

        results = []
        for chunk_json, distance in rows:
            # Convert distance to similarity score
            score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
            if score >= score_threshold:
                try:
                    chunk_data = json.loads(chunk_json)
                    chunk = EmbeddedChunk.model_validate(chunk_data)
                    results.append((chunk, score))
                except (json.JSONDecodeError, ValueError):
                    continue

        return results

    async def _search_json(
        self, query_embedding: list[float], limit: int, score_threshold: float
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Search using JSON embeddings and numpy cosine similarity."""

        def _execute_search():
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute(f"""
                    SELECT chunk_data, embedding_json
                    FROM {self.metadata_table}
                    WHERE embedding_json IS NOT NULL
                """)
                return cursor.fetchall()
            finally:
                conn.close()

        rows = await asyncio.to_thread(_execute_search)

        results = []
        query_vector = np.array(query_embedding, dtype=np.float32)

        for chunk_json, embedding_json in rows:
            try:
                embedding = json.loads(embedding_json)
                chunk_vector = np.array(embedding, dtype=np.float32)

                # Cosine similarity
                similarity = float(
                    np.dot(query_vector, chunk_vector)
                    / (np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector))
                )

                if similarity >= score_threshold:
                    chunk_data = json.loads(chunk_json)
                    chunk = EmbeddedChunk.model_validate(chunk_data)
                    results.append((chunk, similarity))

            except (json.JSONDecodeError, ValueError):
                continue

        # Sort by similarity (descending) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID."""
        if not chunk_ids:
            return 0

        def _execute_deletes():
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute("BEGIN TRANSACTION")
                deleted_count = 0

                # Delete from metadata table
                placeholders = ",".join("?" * len(chunk_ids))
                cursor.execute(
                    f"""
                    DELETE FROM {self.metadata_table}
                    WHERE id IN ({placeholders})
                """,
                    chunk_ids,
                )
                deleted_count = cursor.rowcount

                # Delete from vector table if sqlite-vec is available
                if self._sqlite_vec_available:
                    vector_table = f"{self.table_prefix}_vectors"
                    cursor.execute(
                        f"""
                        DELETE FROM {vector_table}
                        WHERE chunk_id IN ({placeholders})
                    """,
                        chunk_ids,
                    )

                conn.commit()
                return deleted_count
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return await asyncio.to_thread(_execute_deletes)

    async def get_chunk(self, chunk_id: str) -> EmbeddedChunk | None:
        """Get a chunk by ID."""

        def _execute_get():
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute(
                    f"""
                    SELECT chunk_data FROM {self.metadata_table}
                    WHERE id = ?
                """,
                    (chunk_id,),
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

        chunk_json = await asyncio.to_thread(_execute_get)
        if chunk_json is None:
            return None

        try:
            chunk_data = json.loads(chunk_json)
            return EmbeddedChunk.model_validate(chunk_data)
        except (json.JSONDecodeError, ValueError):
            return None
