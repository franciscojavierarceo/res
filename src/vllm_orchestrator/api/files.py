"""Files API router - OpenAI-compatible /v1/files endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from vllm_orchestrator.core.files.service import FilesService
from vllm_orchestrator.core.files.storage import LocalObjectStorage

router = APIRouter(prefix="/v1/files", tags=["files"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class FileObject(BaseModel):
    """Response object for file operations."""

    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "uploaded"


class ListFilesResponse(BaseModel):
    """Response for GET /v1/files."""

    object: str = "list"
    data: list[FileObject]


class DeleteFileResponse(BaseModel):
    """Response for DELETE /v1/files/{id}."""

    id: str
    object: str = "file.deleted"
    deleted: bool


# -----------------------------------------------------------------------------
# Dependency Injection
# -----------------------------------------------------------------------------


def get_files_service(request: Request) -> FilesService:
    """Get FilesService from app state or create one if needed."""
    # For now, create a simple local storage service
    # In production, this would come from proper app state
    if not hasattr(request.app.state, 'files_service'):
        storage = LocalObjectStorage(base_path="/tmp/vllm_orchestrator_files")
        db_manager = request.app.state.db_manager
        request.app.state.files_service = FilesService(
            object_storage=storage,
            db_manager=db_manager,
        )
    return request.app.state.files_service


def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from request."""
    return request.headers.get("X-Tenant-ID", "default")


FilesServiceDep = Annotated[FilesService, Depends(get_files_service)]
TenantIdDep = Annotated[str, Depends(get_tenant_id)]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post("")
async def upload_file(
    service: FilesServiceDep,
    tenant_id: TenantIdDep,
    file: UploadFile = File(...),
    purpose: str = Form(...),
) -> FileObject:
    """Upload a file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Read file content
    content = await file.read()

    try:
        # Upload file
        file_model = await service.upload_file(
            filename=file.filename,
            content=content,
            purpose=purpose,
            tenant_id=tenant_id,
        )

        return FileObject(
            id=file_model.id,
            bytes=file_model.bytes,
            created_at=int(file_model.created_at.timestamp()),
            filename=file_model.filename,
            purpose=file_model.purpose,
            status=file_model.status,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{file_id}")
async def get_file(
    file_id: str,
    service: FilesServiceDep,
    tenant_id: TenantIdDep,
) -> FileObject:
    """Retrieve file metadata."""
    file_model = await service.get_file(file_id, tenant_id)
    if not file_model:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    return FileObject(
        id=file_model.id,
        bytes=file_model.bytes,
        created_at=int(file_model.created_at.timestamp()),
        filename=file_model.filename,
        purpose=file_model.purpose,
        status=file_model.status,
    )


@router.get("")
async def list_files(
    service: FilesServiceDep,
    tenant_id: TenantIdDep,
    purpose: str | None = None,
    limit: int = 20,
) -> ListFilesResponse:
    """List uploaded files."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    files = await service.list_files(
        tenant_id=tenant_id,
        purpose=purpose,
        limit=limit,
    )

    data = [
        FileObject(
            id=f.id,
            bytes=f.bytes,
            created_at=int(f.created_at.timestamp()),
            filename=f.filename,
            purpose=f.purpose,
            status=f.status,
        )
        for f in files
    ]

    return ListFilesResponse(data=data)


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    service: FilesServiceDep,
    tenant_id: TenantIdDep,
) -> DeleteFileResponse:
    """Delete a file."""
    deleted = await service.delete_file(file_id, tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    return DeleteFileResponse(id=file_id, deleted=True)


@router.get("/{file_id}/content")
async def get_file_content(
    file_id: str,
    service: FilesServiceDep,
    tenant_id: TenantIdDep,
):
    """Download file content."""
    content = await service.get_file_content(file_id, tenant_id)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    # Get file metadata for proper response headers
    file_model = await service.get_file(file_id, tenant_id)
    if not file_model:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    from fastapi.responses import Response

    return Response(
        content=content,
        media_type=file_model.mime_type or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file_model.filename}"}
    )