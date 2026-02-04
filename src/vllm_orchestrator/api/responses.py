"""Responses API router - OpenAI-compatible /v1/responses endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from vllm_orchestrator.core.responses import ResponsesService

router = APIRouter(prefix="/v1/responses", tags=["responses"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class CreateResponseRequest(BaseModel):
    """Request body for POST /v1/responses."""

    model: str = Field(description="Model identifier")
    input: str | list[dict[str, Any]] = Field(
        description="Input text or list of input items"
    )
    instructions: str | None = Field(
        default=None,
        description="System instructions",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="ID of previous response for multi-turn conversation",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool definitions",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="Tool choice strategy",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Request metadata",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    store: bool = Field(
        default=True,
        description="Whether to persist the response",
    )


class ResponseObject(BaseModel):
    """Response object returned from GET /v1/responses/{id}."""

    id: str
    object: str = "response"
    created_at: int
    model: str
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    previous_response_id: str | None = None
    metadata: dict[str, str] | None = None


class ListResponsesResponse(BaseModel):
    """Response for GET /v1/responses."""

    object: str = "list"
    data: list[ResponseObject]
    has_more: bool = False


class DeleteResponseResponse(BaseModel):
    """Response for DELETE /v1/responses/{id}."""

    id: str
    object: str = "response.deleted"
    deleted: bool


# -----------------------------------------------------------------------------
# Dependency Injection
# -----------------------------------------------------------------------------


def get_responses_service(request: Request) -> ResponsesService:
    """Get ResponsesService from app state."""
    return request.app.state.responses_service


ResponsesServiceDep = Annotated[ResponsesService, Depends(get_responses_service)]


def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from request.

    In production, this would come from authentication.
    For now, uses a header or defaults to "default".
    """
    return request.headers.get("X-Tenant-ID", "default")


TenantIdDep = Annotated[str, Depends(get_tenant_id)]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post("")
async def create_response(
    request_body: CreateResponseRequest,
    service: ResponsesServiceDep,
    tenant_id: TenantIdDep,
):
    """Create a new response.

    If previous_response_id is provided, loads the conversation history
    and continues the conversation.
    """
    try:
        result = await service.create_response(
            model=request_body.model,
            input=request_body.input,
            instructions=request_body.instructions,
            previous_response_id=request_body.previous_response_id,
            max_output_tokens=request_body.max_output_tokens,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            tools=request_body.tools,
            tool_choice=request_body.tool_choice,
            metadata=request_body.metadata,
            stream=request_body.stream,
            store=request_body.store,
            tenant_id=tenant_id,
        )

        if request_body.stream:
            # Return SSE stream
            async def event_generator():
                async for event in result:
                    if hasattr(event, "model_dump_json"):
                        yield {"event": event.type, "data": event.model_dump_json()}
                    else:
                        yield {"event": "message", "data": str(event)}

            return EventSourceResponse(event_generator())
        else:
            # Return Response object
            return result.model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{response_id}")
async def get_response(
    response_id: str,
    service: ResponsesServiceDep,
    tenant_id: TenantIdDep,
) -> ResponseObject:
    """Retrieve a stored response by ID."""
    response = await service.get_response(response_id, tenant_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

    return ResponseObject(
        id=response.id,
        created_at=int(response.created_at.timestamp()),
        model=response.model,
        status=response.status,
        input=response.input,
        output=response.output,
        usage=response.usage,
        error=response.error,
        previous_response_id=response.previous_response_id,
        metadata=response.metadata_,
    )


@router.get("")
async def list_responses(
    service: ResponsesServiceDep,
    tenant_id: TenantIdDep,
    limit: int = 20,
    after: str | None = None,
    before: str | None = None,
    order: str = "desc",
) -> ListResponsesResponse:
    """List stored responses with pagination."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")

    responses = await service.list_responses(
        tenant_id=tenant_id,
        limit=limit + 1,  # Fetch one extra to check has_more
        after=after,
        before=before,
        order=order,
    )

    has_more = len(responses) > limit
    if has_more:
        responses = responses[:limit]

    data = [
        ResponseObject(
            id=r.id,
            created_at=int(r.created_at.timestamp()),
            model=r.model,
            status=r.status,
            input=r.input,
            output=r.output,
            usage=r.usage,
            error=r.error,
            previous_response_id=r.previous_response_id,
            metadata=r.metadata_,
        )
        for r in responses
    ]

    return ListResponsesResponse(data=data, has_more=has_more)


@router.delete("/{response_id}")
async def delete_response(
    response_id: str,
    service: ResponsesServiceDep,
    tenant_id: TenantIdDep,
) -> DeleteResponseResponse:
    """Delete a stored response."""
    deleted = await service.delete_response(response_id, tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

    return DeleteResponseResponse(id=response_id, deleted=True)


@router.post("/{response_id}/cancel")
async def cancel_response(
    response_id: str,
    service: ResponsesServiceDep,
    tenant_id: TenantIdDep,
) -> ResponseObject:
    """Cancel an in-progress response.

    Note: This is a placeholder. Full cancellation support requires
    integration with vLLM's cancellation mechanism.
    """
    response = await service.get_response(response_id, tenant_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

    if response.status != "in_progress":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel response with status '{response.status}'",
        )

    # TODO: Implement actual cancellation via vLLM
    # For now, just mark as cancelled in storage

    return ResponseObject(
        id=response.id,
        created_at=int(response.created_at.timestamp()),
        model=response.model,
        status="cancelled",
        input=response.input,
        output=response.output,
        usage=response.usage,
        error=response.error,
        previous_response_id=response.previous_response_id,
        metadata=response.metadata_,
    )
