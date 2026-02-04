"""Persistent storage for responses."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vllm_orchestrator.storage.models import ResponseModel


class ResponsesStore:
    """Database operations for responses.

    Handles CRUD operations and response chain resolution.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        response_id: str,
        model: str,
        input_data: dict[str, Any],
        tenant_id: str,
        *,
        status: str = "in_progress",
        previous_response_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResponseModel:
        """Create a new response record.

        Args:
            response_id: Unique response ID (resp_xxx format)
            model: Model identifier
            input_data: Original input (stored as JSON)
            tenant_id: Tenant identifier for multi-tenancy
            status: Response status (in_progress, completed, failed)
            previous_response_id: ID of previous response in chain
            metadata: Optional metadata

        Returns:
            Created ResponseModel instance
        """
        response = ResponseModel(
            id=response_id,
            model=model,
            status=status,
            input=input_data,
            previous_response_id=previous_response_id,
            metadata_=metadata,
            tenant_id=tenant_id,
        )
        self._session.add(response)
        await self._session.flush()
        return response

    async def get(
        self,
        response_id: str,
        tenant_id: str,
    ) -> ResponseModel | None:
        """Get a response by ID.

        Args:
            response_id: Response ID to fetch
            tenant_id: Tenant identifier (for access control)

        Returns:
            ResponseModel if found, None otherwise
        """
        result = await self._session.execute(
            select(ResponseModel).where(
                ResponseModel.id == response_id,
                ResponseModel.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        response_id: str,
        tenant_id: str,
        *,
        status: str | None = None,
        output: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> ResponseModel | None:
        """Update a response record.

        Args:
            response_id: Response ID to update
            tenant_id: Tenant identifier
            status: New status
            output: Response output data
            usage: Token usage data
            error: Error information (if failed)

        Returns:
            Updated ResponseModel if found, None otherwise
        """
        response = await self.get(response_id, tenant_id)
        if response is None:
            return None

        if status is not None:
            response.status = status
        if output is not None:
            response.output = output
        if usage is not None:
            response.usage = usage
        if error is not None:
            response.error = error

        await self._session.flush()
        return response

    async def get_chain(
        self,
        response_id: str,
        tenant_id: str,
        *,
        max_depth: int = 100,
    ) -> list[ResponseModel]:
        """Get the full response chain leading up to a response.

        Follows the previous_response_id links to build conversation history.

        Args:
            response_id: Starting response ID
            tenant_id: Tenant identifier
            max_depth: Maximum chain depth (prevents infinite loops)

        Returns:
            List of responses from oldest to newest (chronological order)
        """
        chain: list[ResponseModel] = []
        current_id: str | None = response_id
        depth = 0

        while current_id is not None and depth < max_depth:
            response = await self.get(current_id, tenant_id)
            if response is None:
                break

            chain.append(response)
            current_id = response.previous_response_id
            depth += 1

        # Reverse to get chronological order (oldest first)
        chain.reverse()
        return chain

    async def list_responses(
        self,
        tenant_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ) -> list[ResponseModel]:
        """List responses with pagination.

        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of responses to return
            after: Cursor for forward pagination (response ID)
            before: Cursor for backward pagination (response ID)
            order: Sort order ("asc" or "desc")

        Returns:
            List of ResponseModel instances
        """
        query = select(ResponseModel).where(ResponseModel.tenant_id == tenant_id)

        # Apply cursor-based pagination
        if after is not None:
            after_response = await self.get(after, tenant_id)
            if after_response:
                query = query.where(
                    ResponseModel.created_at > after_response.created_at
                )

        if before is not None:
            before_response = await self.get(before, tenant_id)
            if before_response:
                query = query.where(
                    ResponseModel.created_at < before_response.created_at
                )

        # Apply ordering
        if order == "desc":
            query = query.order_by(ResponseModel.created_at.desc())
        else:
            query = query.order_by(ResponseModel.created_at.asc())

        query = query.limit(limit)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def delete(
        self,
        response_id: str,
        tenant_id: str,
    ) -> bool:
        """Delete a response.

        Args:
            response_id: Response ID to delete
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        response = await self.get(response_id, tenant_id)
        if response is None:
            return False

        await self._session.delete(response)
        await self._session.flush()
        return True
