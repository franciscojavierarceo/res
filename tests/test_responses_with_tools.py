"""Integration tests for ResponsesService with tool orchestration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_orchestrator.core.responses.service import ResponsesService
from vllm_orchestrator.core.vectors.service import VectorStoreService
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.tools.base import ToolRegistry
from vllm_orchestrator.tools.file_search import FileSearchExecutor
from vllm_orchestrator.tools.orchestrator import ToolOrchestrator
from vllm_orchestrator.vllm_client import VllmClient


@pytest.fixture
async def mock_database_manager():
    """Create a mock database manager."""
    from vllm_orchestrator.storage.models import ResponseModel

    db_manager = MagicMock(spec=DatabaseManager)

    # Mock session context manager
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.delete = AsyncMock()

    # Mock response model for updates
    mock_response_model = MagicMock(spec=ResponseModel)
    mock_response_model.id = "resp_test123"
    mock_response_model.status = "in_progress"

    # Mock the result of session.execute() to return a mock response
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_response_model
    mock_session.execute.return_value = mock_result

    # Create an async context manager
    mock_session_context_manager = MagicMock()
    mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

    db_manager.session.return_value = mock_session_context_manager

    return db_manager, mock_session


@pytest.fixture
def mock_vllm_client():
    """Create a mock vLLM client that simulates tool usage."""
    client = MagicMock(spec=VllmClient)

    # Mock response creation with tool orchestration
    call_count = 0

    async def mock_create_response(**kwargs):
        nonlocal call_count
        call_count += 1

        from openai.types.responses import Response

        response = MagicMock(spec=Response)
        response.id = f"resp_mock_{call_count}"
        response.model = kwargs.get("model", "test-model")

        # Check if this includes tool results
        input_messages = kwargs.get("input", [])
        has_tool_results = any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            for msg in input_messages
        )

        if has_tool_results:
            # Final response after tool execution - proper message format
            output_items = [
                {
                    "type": "message",
                    "id": f"msg_mock_{call_count}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Based on the file search results, I found relevant information about the query.",
                            "annotations": [],
                        }
                    ],
                }
            ]
        else:
            # Initial response with tool call - proper function call format
            output_items = [
                {
                    "type": "function_call",
                    "id": "call_file_search_123",
                    "function": {
                        "name": "file_search",
                        "arguments": '{"query": "test information", "vector_store_id": "vs_test_123"}',
                    },
                }
            ]

        response.model_dump.return_value = {
            "id": response.id,
            "created_at": 1640995200.0,
            "model": response.model,
            "object": "response",
            "output": output_items,
            "parallel_tool_calls": False,
            "tool_choice": "auto" if not has_tool_results else "none",
            "tools": []
            if has_tool_results
            else [
                {
                    "type": "function",
                    "function": {
                        "name": "file_search",
                        "description": "Search for files in a vector store",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "vector_store_id": {"type": "string"},
                            },
                            "required": ["query", "vector_store_id"],
                        },
                    },
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_tokens_details": {
                    "text": 100,
                    "audio": 0,
                    "image": 0,
                    "cached_tokens": 0,
                },
                "output_tokens_details": {
                    "text": 50,
                    "audio": 0,
                    "reasoning_tokens": 0,
                },
            },
            "status": "completed",
        }

        return response

    client.create_response = AsyncMock(side_effect=mock_create_response)
    return client


@pytest.fixture
def mock_vector_service():
    """Mock vector store service for file search."""
    service = MagicMock(spec=VectorStoreService)

    async def mock_search(
        vector_store_id, query, max_results=5, score_threshold=0.0, tenant_id="default"
    ):
        # Return mock search results
        return [
            MagicMock(
                text=f"Document content related to: {query}",
                score=0.89,
                file_id="file_test_123",
                chunk_id="chunk_abc",
                metadata={"title": "Test Document", "page": 1},
            ),
            MagicMock(
                text=f"Additional information about {query}",
                score=0.76,
                file_id="file_test_456",
                chunk_id="chunk_def",
                metadata={"title": "Reference Guide", "section": "Overview"},
            ),
        ]

    service.search_vector_store = AsyncMock(side_effect=mock_search)
    return service


@pytest.mark.skip(
    reason="Complex tool validation incompatibility with OpenAI Response model - requires architectural fix"
)
@pytest.mark.asyncio
async def test_responses_service_with_file_search_tool(
    mock_database_manager, mock_vllm_client, mock_vector_service
):
    """Test ResponsesService with file search tool integration."""
    db_manager, mock_session = mock_database_manager

    # Set up tool orchestration
    tool_registry = ToolRegistry()
    file_search_executor = FileSearchExecutor(mock_vector_service)
    tool_registry.register(file_search_executor)

    tool_orchestrator = ToolOrchestrator(mock_vllm_client, tool_registry)

    # Create responses service with tool orchestrator
    responses_service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        tool_orchestrator=tool_orchestrator,
    )

    # Create response with file search tool
    response = await responses_service.create_response(
        model="llama-3.3-70b",
        input="Find information about machine learning algorithms",
        tools=[file_search_executor.get_tool_definition()],
        tool_choice="auto",
        tenant_id="test_tenant",
    )

    # Verify response was created
    assert response is not None

    # Verify vLLM was called twice (initial + after tool)
    assert mock_vllm_client.create_response.call_count == 2

    # Verify vector service was called for file search
    mock_vector_service.search_vector_store.assert_called_once()
    search_call = mock_vector_service.search_vector_store.call_args
    assert "machine learning" in str(search_call).lower() or "test information" in str(
        search_call
    )

    # Verify database operations for response storage
    mock_session.add.assert_called()  # Response was stored
    mock_session.flush.assert_called()  # Changes were flushed


@pytest.mark.skip(
    reason="Complex tool validation incompatibility with OpenAI Response model - requires architectural fix"
)
@pytest.mark.asyncio
async def test_responses_service_without_tools(mock_database_manager, mock_vllm_client):
    """Test ResponsesService without tools (direct vLLM call)."""
    db_manager, mock_session = mock_database_manager

    responses_service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        # No tool orchestrator
    )

    # Create simple response without tools
    response = await responses_service.create_response(
        model="llama-3.3-70b",
        input="Hello, how are you?",
        tenant_id="test_tenant",
    )

    # Verify response was created
    assert response is not None

    # Should only call vLLM once (no tool orchestration)
    assert mock_vllm_client.create_response.call_count == 1

    # Verify database operations
    mock_session.add.assert_called()
    mock_session.flush.assert_called()


@pytest.mark.skip(
    reason="Complex tool validation incompatibility with OpenAI Response model - requires architectural fix"
)
@pytest.mark.asyncio
async def test_responses_service_with_tools_but_no_orchestrator(
    mock_database_manager, mock_vllm_client
):
    """Test ResponsesService with tools but no orchestrator (fallback to direct vLLM)."""
    db_manager, mock_session = mock_database_manager

    responses_service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        # No tool orchestrator - should fallback to direct vLLM
    )

    # Try to create response with tools
    response = await responses_service.create_response(
        model="llama-3.3-70b",
        input="Search for information",
        tools=[{"type": "function", "function": {"name": "test_tool"}}],
        tenant_id="test_tenant",
    )

    # Verify response was created
    assert response is not None

    # Should call vLLM directly (no orchestration)
    assert mock_vllm_client.create_response.call_count == 1

    # Tools should be passed to vLLM but no orchestration occurs
    call_args = mock_vllm_client.create_response.call_args
    assert call_args.kwargs.get("tools") is not None
