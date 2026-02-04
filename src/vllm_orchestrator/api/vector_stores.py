"""Vector Stores API router - OpenAI-compatible /v1/vector_stores endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from vllm_orchestrator.core.vectors.service import VectorStoreService
from vllm_orchestrator.core.files.service import FilesService
from vllm_orchestrator.core.files.storage import LocalObjectStorage

router = APIRouter(prefix="/v1/vector_stores", tags=["vector_stores"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class CreateVectorStoreRequest(BaseModel):
    """Request body for POST /v1/vector_stores."""

    name: str = Field(description="Name of the vector store")
    file_ids: list[str] | None = Field(
        default=None,
        description="List of file IDs to add to the vector store",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional metadata",
    )
    expires_after: dict[str, Any] | None = Field(
        default=None,
        description="Expiration configuration",
    )


class VectorStoreObject(BaseModel):
    """Response object for vector store operations."""

    id: str
    object: str = "vector_store"
    name: str
    status: str
    usage_bytes: int = 0
    file_counts: dict[str, int]
    created_at: int
    metadata: dict[str, str] | None = None
    expires_after: dict[str, Any] | None = None
    expires_at: int | None = None
    last_active_at: int | None = None


class ListVectorStoresResponse(BaseModel):
    """Response for GET /v1/vector_stores."""

    object: str = "list"
    data: list[VectorStoreObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class DeleteVectorStoreResponse(BaseModel):
    """Response for DELETE /v1/vector_stores/{id}."""

    id: str
    object: str = "vector_store.deleted"
    deleted: bool


class AddFileToVectorStoreRequest(BaseModel):
    """Request body for POST /v1/vector_stores/{id}/files."""

    file_id: str = Field(description="ID of the file to add")


class VectorStoreFileObject(BaseModel):
    """Response object for vector store file operations."""

    id: str
    object: str = "vector_store.file"
    usage_bytes: int = 0
    created_at: int
    vector_store_id: str
    status: str
    last_error: dict[str, Any] | None = None


class ListVectorStoreFilesResponse(BaseModel):
    """Response for GET /v1/vector_stores/{id}/files."""

    object: str = "list"
    data: list[VectorStoreFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class DeleteVectorStoreFileResponse(BaseModel):
    """Response for DELETE /v1/vector_stores/{id}/files/{file_id}."""

    id: str
    object: str = "vector_store.file.deleted"
    deleted: bool


class SearchVectorStoreRequest(BaseModel):
    """Request body for POST /v1/vector_stores/{id}/search."""

    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=100, description="Maximum results")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class SearchResult(BaseModel):
    """Individual search result."""

    id: str
    score: float
    content: str
    metadata: dict[str, Any] | None = None


class SearchVectorStoreResponse(BaseModel):
    """Response for POST /v1/vector_stores/{id}/search."""

    object: str = "list"
    data: list[SearchResult]


# -----------------------------------------------------------------------------
# Dependency Injection
# -----------------------------------------------------------------------------


def get_vector_store_service(request: Request) -> VectorStoreService:
    """Get VectorStoreService from app state or create one if needed."""
    if not hasattr(request.app.state, 'vector_store_service'):
        # Create files service first
        storage = LocalObjectStorage(base_path="/tmp/vllm_orchestrator_files")
        db_manager = request.app.state.db_manager
        vllm_client = request.app.state.vllm_client
        files_service = FilesService(object_storage=storage, db_manager=db_manager)

        request.app.state.vector_store_service = VectorStoreService(
            db_manager=db_manager,
            vllm_client=vllm_client,
            files_service=files_service,
            default_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    return request.app.state.vector_store_service


def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from request."""
    return request.headers.get("X-Tenant-ID", "default")


VectorStoreServiceDep = Annotated[VectorStoreService, Depends(get_vector_store_service)]
TenantIdDep = Annotated[str, Depends(get_tenant_id)]


# -----------------------------------------------------------------------------
# Vector Store Endpoints
# -----------------------------------------------------------------------------


@router.post("")
async def create_vector_store(
    request_body: CreateVectorStoreRequest,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> VectorStoreObject:
    """Create a new vector store."""
    try:
        vector_store = await service.create_vector_store(
            name=request_body.name,
            file_ids=request_body.file_ids or [],
            metadata=request_body.metadata,
            tenant_id=tenant_id,
        )

        return VectorStoreObject(
            id=vector_store.id,
            name=vector_store.name,
            status=vector_store.status,
            file_counts=vector_store.file_counts,
            created_at=int(vector_store.created_at.timestamp()),
            metadata=vector_store.metadata_,
            expires_at=int(vector_store.expires_at.timestamp()) if vector_store.expires_at else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")


@router.get("/{vector_store_id}")
async def get_vector_store(
    vector_store_id: str,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> VectorStoreObject:
    """Retrieve a vector store."""
    vector_store = await service.get_vector_store(vector_store_id, tenant_id)
    if not vector_store:
        raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

    return VectorStoreObject(
        id=vector_store.id,
        name=vector_store.name,
        status=vector_store.status,
        file_counts=vector_store.file_counts,
        created_at=int(vector_store.created_at.timestamp()),
        metadata=vector_store.metadata_,
        expires_at=int(vector_store.expires_at.timestamp()) if vector_store.expires_at else None,
    )


@router.get("")
async def list_vector_stores(
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
    limit: int = 20,
    order: str = "desc",
    after: str | None = None,
    before: str | None = None,
) -> ListVectorStoresResponse:
    """List vector stores."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")

    vector_stores = await service.list_vector_stores(
        tenant_id=tenant_id,
        limit=limit + 1,  # Fetch one extra to check has_more
        order=order,
        after=after,
        before=before,
    )

    has_more = len(vector_stores) > limit
    if has_more:
        vector_stores = vector_stores[:limit]

    data = [
        VectorStoreObject(
            id=vs.id,
            name=vs.name,
            status=vs.status,
            file_counts=vs.file_counts,
            created_at=int(vs.created_at.timestamp()),
            metadata=vs.metadata_,
            expires_at=int(vs.expires_at.timestamp()) if vs.expires_at else None,
        )
        for vs in vector_stores
    ]

    return ListVectorStoresResponse(
        data=data,
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
        has_more=has_more,
    )


@router.patch("/{vector_store_id}")
async def update_vector_store(
    vector_store_id: str,
    request_body: dict[str, Any],
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> VectorStoreObject:
    """Update a vector store."""
    try:
        vector_store = await service.update_vector_store(
            vector_store_id=vector_store_id,
            tenant_id=tenant_id,
            **request_body,
        )

        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        return VectorStoreObject(
            id=vector_store.id,
            name=vector_store.name,
            status=vector_store.status,
            file_counts=vector_store.file_counts,
            created_at=int(vector_store.created_at.timestamp()),
            metadata=vector_store.metadata_,
            expires_at=int(vector_store.expires_at.timestamp()) if vector_store.expires_at else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update vector store: {str(e)}")


@router.delete("/{vector_store_id}")
async def delete_vector_store(
    vector_store_id: str,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> DeleteVectorStoreResponse:
    """Delete a vector store."""
    deleted = await service.delete_vector_store(vector_store_id, tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

    return DeleteVectorStoreResponse(id=vector_store_id, deleted=True)


# -----------------------------------------------------------------------------
# Vector Store Files Endpoints
# -----------------------------------------------------------------------------


@router.post("/{vector_store_id}/files")
async def add_file_to_vector_store(
    vector_store_id: str,
    request_body: AddFileToVectorStoreRequest,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> VectorStoreFileObject:
    """Add a file to a vector store."""
    try:
        result = await service.add_file_to_vector_store(
            vector_store_id=vector_store_id,
            file_id=request_body.file_id,
            tenant_id=tenant_id,
        )

        # For simplicity, return a basic file object
        # In a real implementation, you'd track processing status
        import time
        return VectorStoreFileObject(
            id=request_body.file_id,
            created_at=int(time.time()),
            vector_store_id=vector_store_id,
            status="completed" if result else "failed",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add file: {str(e)}")


@router.get("/{vector_store_id}/files")
async def list_vector_store_files(
    vector_store_id: str,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
    limit: int = 20,
    order: str = "desc",
    after: str | None = None,
    before: str | None = None,
) -> ListVectorStoreFilesResponse:
    """List files in a vector store."""
    # This would need to be implemented in the service
    # For now, return empty list
    return ListVectorStoreFilesResponse(data=[])


@router.delete("/{vector_store_id}/files/{file_id}")
async def remove_file_from_vector_store(
    vector_store_id: str,
    file_id: str,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> DeleteVectorStoreFileResponse:
    """Remove a file from a vector store."""
    try:
        deleted = await service.remove_file_from_vector_store(
            vector_store_id=vector_store_id,
            file_id=file_id,
            tenant_id=tenant_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found in vector store {vector_store_id}"
            )

        return DeleteVectorStoreFileResponse(id=file_id, deleted=True)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove file: {str(e)}")


# -----------------------------------------------------------------------------
# Search Endpoint
# -----------------------------------------------------------------------------


@router.post("/{vector_store_id}/search")
async def search_vector_store(
    vector_store_id: str,
    request_body: SearchVectorStoreRequest,
    service: VectorStoreServiceDep,
    tenant_id: TenantIdDep,
) -> SearchVectorStoreResponse:
    """Search a vector store."""
    try:
        results = await service.search(
            vector_store_id=vector_store_id,
            query=request_body.query,
            max_results=request_body.max_results,
            score_threshold=request_body.score_threshold,
            tenant_id=tenant_id,
        )

        data = [
            SearchResult(
                id=result.chunk_id,
                score=result.score,
                content=result.text,
                metadata=result.metadata,
            )
            for result in results
        ]

        return SearchVectorStoreResponse(data=data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")