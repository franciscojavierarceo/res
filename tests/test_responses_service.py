"""Tests for responses service."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import Response, ResponseInputItemParam

from vllm_orchestrator.core.responses.service import (
    ResponsesService,
    generate_response_id,
)
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.vllm_client import VllmClient


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
async def mock_database_manager():
    """Create a mock database manager."""
    db_manager = MagicMock(spec=DatabaseManager)

    # Mock session context manager
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.delete = AsyncMock()

    # Mock the result of session.execute() for database queries
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None  # Default to no results
    mock_result.scalars.return_value.all.return_value = []  # Default to empty list
    mock_session.execute.return_value = mock_result

    # Create an async context manager
    mock_session_context_manager = MagicMock()
    mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

    db_manager.session.return_value = mock_session_context_manager

    return db_manager, mock_session


@pytest.fixture
def mock_vllm_client():
    """Create a mock vLLM client."""
    client = MagicMock(spec=VllmClient)

    # Mock response creation
    async def mock_create_response(**kwargs):
        # Create a proper response dict with ALL required fields
        response_data = {
            "id": "resp_mock123",
            "created_at": 1640995200.0,
            "model": kwargs.get("model", "test-model"),
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "id": "msg_mock123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Mock response content",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            # Optional fields
            "usage": {
                "input_tokens": 30,
                "output_tokens": 20,
                "total_tokens": 50,
                "input_tokens_details": {
                    "text": 30,
                    "audio": 0,
                    "image": 0,
                    "cached_tokens": 0,
                },
                "output_tokens_details": {
                    "text": 20,
                    "audio": 0,
                    "reasoning_tokens": 0,
                },
            },
            "status": "completed",
            "metadata": None,
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
        }

        # Create a mock response that returns proper data when model_dump() is called
        response = MagicMock(spec=Response)
        response.id = response_data["id"]
        response.model = response_data["model"]
        response.output = response_data["output"]
        response.usage = response_data["usage"]
        response.status = response_data["status"]
        response.model_dump.return_value = response_data

        return response

    client.create_response = AsyncMock(side_effect=mock_create_response)
    return client


def test_generate_response_id():
    """Test response ID generation."""
    response_id = generate_response_id()

    assert response_id.startswith("resp_")
    assert len(response_id) == 29  # "resp_" + 24 hex chars

    # Generate another to ensure uniqueness
    response_id2 = generate_response_id()
    assert response_id != response_id2


@pytest.mark.asyncio
async def test_responses_service_initialization(
    mock_database_manager, mock_vllm_client
):
    """Test ResponsesService initialization."""
    db_manager, _ = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        default_tenant_id="test_tenant",
    )

    assert service._vllm_client == mock_vllm_client
    assert service._db_manager == db_manager
    assert service._default_tenant_id == "test_tenant"


@pytest.mark.asyncio
async def test_create_simple_response(mock_database_manager, mock_vllm_client):
    """Test creating a simple response without previous context."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Create response
    response = await service.create_response(
        model="test-model",
        input="Hello, how are you?",
        max_output_tokens=100,
        temperature=0.7,
    )

    # Verify vLLM client was called
    mock_vllm_client.create_response.assert_called_once()
    call_args = mock_vllm_client.create_response.call_args

    # Verify correct parameters were passed to vLLM client
    assert call_args.kwargs.get("model") == "test-model"
    assert call_args.kwargs.get("max_output_tokens") == 100
    assert call_args.kwargs.get("temperature") == 0.7

    # Verify database operations
    mock_session.add.assert_called()
    mock_session.flush.assert_called()


@pytest.mark.asyncio
async def test_create_response_with_previous_context(
    mock_database_manager, mock_vllm_client
):
    """Test creating a response with previous response context."""
    db_manager, mock_session = mock_database_manager

    # Mock previous response lookup
    mock_previous_response = MagicMock()
    mock_previous_response.id = "resp_previous123"
    mock_previous_response.input = [{"type": "message", "content": "Previous input"}]
    mock_previous_response.output = [{"type": "message", "content": "Previous output"}]

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_previous_response
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Create response with previous context
    response = await service.create_response(
        model="test-model",
        input="Follow up question",
        previous_response_id="resp_previous123",
    )

    # Verify previous response was looked up
    mock_session.execute.assert_called()

    # Verify vLLM client was called with expanded context
    mock_vllm_client.create_response.assert_called_once()
    call_args = mock_vllm_client.create_response.call_args

    # The input should be expanded with previous context
    full_input = call_args.kwargs.get("input")
    assert isinstance(full_input, list)
    assert len(full_input) > 1  # Should include both previous and new input


@pytest.mark.asyncio
async def test_create_response_with_tools(mock_database_manager, mock_vllm_client):
    """Test creating a response with tool definitions."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    # Create response with tools
    response = await service.create_response(
        model="test-model",
        input="Search for vLLM benchmarks",
        tools=tools,
        tool_choice="auto",
    )

    # Verify tools were passed to vLLM
    call_args = mock_vllm_client.create_response.call_args
    assert call_args.kwargs.get("tools") == tools
    assert call_args.kwargs.get("tool_choice") == "auto"


@pytest.mark.asyncio
async def test_create_response_no_store(mock_database_manager, mock_vllm_client):
    """Test creating a response without storing it."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Create response without storing
    response = await service.create_response(
        model="test-model",
        input="Temporary request",
        store=False,
    )

    # Verify vLLM client was called
    mock_vllm_client.create_response.assert_called_once()

    # Verify no database operations occurred
    mock_session.add.assert_not_called()
    mock_session.flush.assert_not_called()


@pytest.mark.asyncio
async def test_create_streaming_response(mock_database_manager, mock_vllm_client):
    """Test creating a streaming response."""
    db_manager, mock_session = mock_database_manager

    # Mock streaming response - need to return async generator directly
    async def mock_create_response_streaming(**kwargs):
        yield {"type": "response.text.delta", "delta": "Hello"}
        yield {"type": "response.text.delta", "delta": " world"}
        yield {"type": "response.done"}

    mock_vllm_client.create_response = AsyncMock(
        side_effect=mock_create_response_streaming
    )

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Create streaming response
    stream = await service.create_response(
        model="test-model",
        input="Stream me a response",
        stream=True,
    )

    # Verify we get an async iterator
    assert hasattr(stream, "__aiter__")

    # Collect stream events
    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 3
    assert events[0]["type"] == "response.text.delta"
    assert events[-1]["type"] == "response.done"


@pytest.mark.asyncio
async def test_get_response(mock_database_manager, mock_vllm_client):
    """Test retrieving a stored response."""
    db_manager, mock_session = mock_database_manager

    # Mock response lookup
    mock_response = MagicMock()
    mock_response.id = "resp_test123"
    mock_response.model = "test-model"
    mock_response.status = "completed"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_response
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Get response
    response = await service.get_response("resp_test123")

    # Verify database lookup
    mock_session.execute.assert_called_once()

    # Verify response returned correctly
    assert response is mock_response
    assert response.id == "resp_test123"


@pytest.mark.asyncio
async def test_get_nonexistent_response(mock_database_manager, mock_vllm_client):
    """Test retrieving a response that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Try to get nonexistent response
    response = await service.get_response("nonexistent")

    assert response is None


@pytest.mark.asyncio
async def test_list_responses(mock_database_manager, mock_vllm_client):
    """Test listing responses."""
    db_manager, mock_session = mock_database_manager

    # Mock response list
    mock_responses = [MagicMock(), MagicMock()]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_responses
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # List responses
    responses = await service.list_responses(limit=10)

    assert responses == mock_responses
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_response(mock_database_manager, mock_vllm_client):
    """Test deleting a response."""
    db_manager, mock_session = mock_database_manager

    # Mock response lookup
    mock_response = MagicMock()
    mock_response.id = "resp_test123"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_response
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Delete response
    result = await service.delete_response("resp_test123")

    assert result is True
    mock_session.delete.assert_called_once()
    mock_session.flush.assert_called()


@pytest.mark.asyncio
async def test_delete_nonexistent_response(mock_database_manager, mock_vllm_client):
    """Test deleting a response that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Try to delete nonexistent response
    result = await service.delete_response("nonexistent")

    assert result is False
    mock_session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_response_with_metadata(mock_database_manager, mock_vllm_client):
    """Test creating a response with custom metadata."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    metadata = {"source": "api", "user_id": "user123", "session": "session456"}

    # Create response with metadata
    response = await service.create_response(
        model="test-model",
        input="Request with metadata",
        metadata=metadata,
    )

    # Verify metadata was passed to vLLM
    call_args = mock_vllm_client.create_response.call_args
    assert call_args.kwargs.get("metadata") == metadata


@pytest.mark.asyncio
async def test_multi_tenant_isolation(mock_database_manager, mock_vllm_client):
    """Test that responses are isolated by tenant."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        default_tenant_id="tenant1",
    )

    # Create response for tenant1
    response1 = await service.create_response(
        model="test-model",
        input="Tenant 1 request",
        tenant_id="tenant1",
    )

    # Create response for tenant2
    response2 = await service.create_response(
        model="test-model",
        input="Tenant 2 request",
        tenant_id="tenant2",
    )

    # Both should call vLLM but with different tenant contexts
    assert mock_vllm_client.create_response.call_count == 2

    # Database operations should be called for both
    assert mock_session.add.call_count == 2
    assert mock_session.flush.call_count == 2
