"""SQLAlchemy ORM models for vLLM Orchestrator."""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


# -----------------------------------------------------------------------------
# Responses
# -----------------------------------------------------------------------------


class ResponseModel(Base):
    """Persistent storage for responses."""

    __tablename__ = "responses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="in_progress"
    )
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    previous_response_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("responses.id"), nullable=True
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Self-referential relationship for response chains
    previous_response: Mapped["ResponseModel | None"] = relationship(
        "ResponseModel", remote_side=[id], backref="next_responses"
    )


# -----------------------------------------------------------------------------
# Conversations
# -----------------------------------------------------------------------------


class ConversationModel(Base):
    """Conversation container."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    items: Mapped[list["ConversationItemModel"]] = relationship(
        "ConversationItemModel",
        back_populates="conversation",
        order_by="ConversationItemModel.sequence_number",
    )


class ConversationItemModel(Base):
    """Individual items within a conversation."""

    __tablename__ = "conversation_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="items"
    )


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------


class FileModel(Base):
    """Uploaded file metadata."""

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


# -----------------------------------------------------------------------------
# Vector Stores
# -----------------------------------------------------------------------------


class VectorStoreModel(Base):
    """Vector store for file search."""

    __tablename__ = "vector_stores"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    embedding_model: Mapped[str] = mapped_column(String(256), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    file_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    chunks: Mapped[list["VectorStoreChunkModel"]] = relationship(
        "VectorStoreChunkModel", back_populates="vector_store"
    )


class VectorStoreChunkModel(Base):
    """Individual chunks with embeddings."""

    __tablename__ = "vector_store_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    vector_store_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("vector_stores.id"), nullable=False, index=True
    )
    file_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("files.id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Note: embedding column is added via migration for pgvector
    # For SQLite, we store as JSON
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    vector_store: Mapped["VectorStoreModel"] = relationship(
        "VectorStoreModel", back_populates="chunks"
    )


# -----------------------------------------------------------------------------
# Prompt Templates (Future)
# -----------------------------------------------------------------------------


class PromptTemplateModel(Base):
    """Prompt templates with variable substitution."""

    __tablename__ = "prompt_templates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
