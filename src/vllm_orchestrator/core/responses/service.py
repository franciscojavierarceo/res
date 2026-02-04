"""Responses service - orchestrates stateful response generation."""

from collections.abc import AsyncIterator
from typing import Any, Optional

from openai.types.responses import Response
from openai.types.responses.response_input_item_param import ResponseInputItemParam

from vllm_orchestrator.core.responses.store import ResponsesStore
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.storage.models import ResponseModel
from vllm_orchestrator.tools.orchestrator import ToolOrchestrator
from vllm_orchestrator.vllm_client import VllmClient


def generate_response_id() -> str:
    """Generate a unique response ID."""
    import uuid

    return f"resp_{uuid.uuid4().hex[:24]}"


class ResponsesService:
    """Orchestrates stateful response generation.

    This service:
    1. Resolves previous_response_id chains from storage
    2. Constructs full conversation context
    3. Forwards to vLLM for inference (with store=False)
    4. Persists responses to database
    5. Handles streaming and tool orchestration
    """

    def __init__(
        self,
        vllm_client: VllmClient,
        db_manager: DatabaseManager,
        default_tenant_id: str = "default",
        tool_orchestrator: Optional[ToolOrchestrator] = None,
    ):
        self._vllm_client = vllm_client
        self._db_manager = db_manager
        self._default_tenant_id = default_tenant_id
        self._tool_orchestrator = tool_orchestrator

    async def create_response(
        self,
        model: str,
        input: str | list[ResponseInputItemParam],
        *,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        stream: bool = False,
        store: bool = True,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> Response | AsyncIterator[Any]:
        """Create a new response.

        If previous_response_id is provided, loads the conversation chain
        and constructs the full context before calling vLLM.

        Args:
            model: Model identifier
            input: User input (string or list of input items)
            instructions: System instructions
            previous_response_id: ID of previous response for multi-turn
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: Tool definitions
            tool_choice: Tool choice strategy
            metadata: Request metadata
            stream: Whether to stream the response
            store: Whether to persist the response (default True)
            tenant_id: Tenant identifier
            **kwargs: Additional parameters

        Returns:
            Response object or async iterator of streaming events
        """
        tenant_id = tenant_id or self._default_tenant_id
        response_id = generate_response_id()

        # Build the full input context
        full_input = await self._build_context(
            input=input,
            previous_response_id=previous_response_id,
            tenant_id=tenant_id,
        )

        # Store the response record (in_progress)
        if store:
            async with self._db_manager.session() as session:
                responses_store = ResponsesStore(session)
                await responses_store.create(
                    response_id=response_id,
                    model=model,
                    input_data=self._serialize_input(input),
                    tenant_id=tenant_id,
                    status="in_progress",
                    previous_response_id=previous_response_id,
                    metadata=metadata,
                )

        try:
            if stream:
                return self._stream_response(
                    response_id=response_id,
                    model=model,
                    full_input=full_input,
                    instructions=instructions,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    tool_choice=tool_choice,
                    metadata=metadata,
                    store=store,
                    tenant_id=tenant_id,
                    **kwargs,
                )
            else:
                return await self._create_response_sync(
                    response_id=response_id,
                    model=model,
                    full_input=full_input,
                    instructions=instructions,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    tool_choice=tool_choice,
                    metadata=metadata,
                    store=store,
                    tenant_id=tenant_id,
                    **kwargs,
                )

        except Exception as e:
            # Update response status to failed
            if store:
                async with self._db_manager.session() as session:
                    responses_store = ResponsesStore(session)
                    await responses_store.update(
                        response_id=response_id,
                        tenant_id=tenant_id,
                        status="failed",
                        error={"message": str(e), "type": type(e).__name__},
                    )
            raise

    async def _create_response_sync(
        self,
        response_id: str,
        model: str,
        full_input: list[ResponseInputItemParam],
        *,
        instructions: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        metadata: dict[str, str] | None,
        store: bool,
        tenant_id: str,
        **kwargs: Any,
    ) -> Response:
        """Create response synchronously (non-streaming)."""
        # Use tool orchestration if tools are specified and orchestrator is available
        if tools and self._tool_orchestrator is not None:
            vllm_response = await self._tool_orchestrator.create_response_with_tools(
                model=model,
                input=full_input,
                instructions=instructions,
                tools=tools,
                tool_choice=tool_choice,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                metadata=metadata,
                stream=False,
                tenant_id=tenant_id,
                **kwargs,
            )
        else:
            # Call vLLM directly (no tools or no orchestrator)
            vllm_response = await self._vllm_client.create_response(
                model=model,
                input=full_input,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                tool_choice=tool_choice,
                metadata=metadata,
                stream=False,
                **kwargs,
            )

        # vLLM returns a Response object
        assert isinstance(vllm_response, Response)

        # Override the ID with our tracked ID
        response_dict = vllm_response.model_dump()
        response_dict["id"] = response_id

        # Update stored response with output
        if store:
            async with self._db_manager.session() as session:
                responses_store = ResponsesStore(session)
                await responses_store.update(
                    response_id=response_id,
                    tenant_id=tenant_id,
                    status="completed",
                    output={"items": response_dict.get("output", [])},
                    usage=response_dict.get("usage"),
                )

        # Reconstruct Response with our ID
        return Response.model_validate(response_dict)

    async def _stream_response(
        self,
        response_id: str,
        model: str,
        full_input: list[ResponseInputItemParam],
        *,
        instructions: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        metadata: dict[str, str] | None,
        store: bool,
        tenant_id: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream response events."""
        output_items: list[dict[str, Any]] = []
        usage_data: dict[str, Any] | None = None
        final_status = "completed"

        try:
            # Use tool orchestration if tools are specified and orchestrator is available
            if tools and self._tool_orchestrator is not None:
                stream = await self._tool_orchestrator.create_response_with_tools(
                    model=model,
                    input=full_input,
                    instructions=instructions,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    metadata=metadata,
                    stream=True,
                    tenant_id=tenant_id,
                    **kwargs,
                )
            else:
                # Call vLLM directly (no tools or no orchestrator)
                stream = await self._vllm_client.create_response(
                    model=model,
                    input=full_input,
                    instructions=instructions,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    tool_choice=tool_choice,
                    metadata=metadata,
                    stream=True,
                    **kwargs,
                )

            async for event in stream:
                # Intercept and modify IDs in events
                event_dict = (
                    event.model_dump() if hasattr(event, "model_dump") else event
                )

                # Track output items and usage from completed events
                if hasattr(event, "type"):
                    if event.type == "response.output_item.done":
                        output_items.append(event_dict.get("item", {}))
                    elif event.type == "response.completed":
                        if hasattr(event, "response"):
                            usage_data = (
                                event.response.usage.model_dump()
                                if event.response.usage
                                else None
                            )

                yield event

        except Exception:
            final_status = "failed"
            raise
        finally:
            # Update stored response
            if store:
                async with self._db_manager.session() as session:
                    responses_store = ResponsesStore(session)
                    await responses_store.update(
                        response_id=response_id,
                        tenant_id=tenant_id,
                        status=final_status,
                        output={"items": output_items} if output_items else None,
                        usage=usage_data,
                    )

    async def _build_context(
        self,
        input: str | list[ResponseInputItemParam],
        previous_response_id: str | None,
        tenant_id: str,
    ) -> list[ResponseInputItemParam]:
        """Build full conversation context from response chain.

        Args:
            input: New user input
            previous_response_id: Previous response to chain from
            tenant_id: Tenant identifier

        Returns:
            List of input items including full conversation history
        """
        context: list[ResponseInputItemParam] = []

        # Load previous response chain
        if previous_response_id is not None:
            async with self._db_manager.session() as session:
                responses_store = ResponsesStore(session)
                chain = await responses_store.get_chain(previous_response_id, tenant_id)

                for response in chain:
                    # Add the input from each response
                    if response.input:
                        input_data = response.input
                        if isinstance(input_data, dict) and "content" in input_data:
                            # Stored as {"content": ...}
                            context.append(input_data)
                        elif isinstance(input_data, list):
                            context.extend(input_data)

                    # Add the output from each response
                    if response.output and "items" in response.output:
                        for item in response.output["items"]:
                            context.append(item)

        # Add the new input
        if isinstance(input, str):
            context.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": input,
                }
            )
        elif isinstance(input, list):
            context.extend(input)
        else:
            context.append(input)

        return context

    def _serialize_input(
        self, input: str | list[ResponseInputItemParam]
    ) -> dict[str, Any]:
        """Serialize input for storage."""
        if isinstance(input, str):
            return {"type": "text", "content": input}
        elif isinstance(input, list):
            return {
                "type": "list",
                "items": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in input
                ],
            }
        else:
            return {"type": "unknown", "content": str(input)}

    async def get_response(
        self,
        response_id: str,
        tenant_id: str | None = None,
    ) -> ResponseModel | None:
        """Get a stored response by ID.

        Args:
            response_id: Response ID
            tenant_id: Tenant identifier

        Returns:
            ResponseModel if found, None otherwise
        """
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            responses_store = ResponsesStore(session)
            return await responses_store.get(response_id, tenant_id)

    async def list_responses(
        self,
        tenant_id: str | None = None,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ) -> list[ResponseModel]:
        """List stored responses.

        Args:
            tenant_id: Tenant identifier
            limit: Maximum responses to return
            after: Pagination cursor (after this ID)
            before: Pagination cursor (before this ID)
            order: Sort order

        Returns:
            List of ResponseModel instances
        """
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            responses_store = ResponsesStore(session)
            return await responses_store.list_responses(
                tenant_id=tenant_id,
                limit=limit,
                after=after,
                before=before,
                order=order,
            )

    async def delete_response(
        self,
        response_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """Delete a stored response.

        Args:
            response_id: Response ID to delete
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        tenant_id = tenant_id or self._default_tenant_id
        async with self._db_manager.session() as session:
            responses_store = ResponsesStore(session)
            return await responses_store.delete(response_id, tenant_id)
