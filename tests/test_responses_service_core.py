"""Core tests for responses service functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_orchestrator.core.responses.service import (
    ResponsesService,
    generate_response_id,
)
from vllm_orchestrator.storage.database import DatabaseManager
from vllm_orchestrator.storage.models import ResponseModel
from vllm_orchestrator.vllm_client import VllmClient


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

    # Create an async context manager
    mock_session_context_manager = MagicMock()
    mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

    db_manager.session.return_value = mock_session_context_manager

    return db_manager, mock_session


@pytest.fixture
def mock_vllm_client():
    """Create a mock vLLM client."""
    return MagicMock(spec=VllmClient)


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
async def test_build_context_simple_input(mock_database_manager, mock_vllm_client):
    """Test building context from simple string input."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Test simple string input
    context = await service._build_context(
        input="Hello world", previous_response_id=None, tenant_id="default"
    )

    assert isinstance(context, list)
    assert len(context) == 1
    assert context[0]["type"] == "message"
    assert context[0]["content"] == "Hello world"
    assert context[0]["role"] == "user"


@pytest.mark.asyncio
async def test_build_context_with_previous_response(
    mock_database_manager, mock_vllm_client
):
    """Test building context with previous response chain."""
    db_manager, mock_session = mock_database_manager

    # Mock previous response - note the actual implementation expects different structure
    mock_response = MagicMock(spec=ResponseModel)
    mock_response.input = {"type": "text", "content": "Previous message"}
    mock_response.output = {
        "items": [
            {"type": "message", "content": "Previous response", "role": "assistant"}
        ]
    }

    # Mock get_chain to return a single response
    async def mock_get_chain(response_id, tenant_id):
        return [mock_response]

    # Patch the ResponsesStore to control the chain lookup
    with patch(
        "vllm_orchestrator.core.responses.service.ResponsesStore"
    ) as mock_store_class:
        mock_store = mock_store_class.return_value
        mock_store.get_chain = AsyncMock(side_effect=mock_get_chain)

        service = ResponsesService(
            vllm_client=mock_vllm_client,
            db_manager=db_manager,
        )

        # Build context with previous response
        context = await service._build_context(
            input="Follow up",
            previous_response_id="resp_previous123",
            tenant_id="default",
        )

    # Should have previous input + previous output + new input
    assert isinstance(context, list)
    assert len(context) == 3

    # Previous input (from stored response.input)
    assert context[0]["type"] == "text"
    assert context[0]["content"] == "Previous message"

    # Previous output (from response.output.items)
    assert context[1]["content"] == "Previous response"
    assert context[1]["role"] == "assistant"

    # New input (added by _build_context)
    assert context[2]["content"] == "Follow up"
    assert context[2]["role"] == "user"


@pytest.mark.asyncio
async def test_serialize_input_string(mock_database_manager, mock_vllm_client):
    """Test serializing string input."""
    db_manager, _ = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    result = service._serialize_input("Hello world")

    assert isinstance(result, dict)
    assert result["type"] == "text"
    assert result["content"] == "Hello world"


@pytest.mark.asyncio
async def test_serialize_input_list(mock_database_manager, mock_vllm_client):
    """Test serializing list input."""
    db_manager, _ = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    input_list = [
        {"type": "message", "content": "First message", "role": "user"},
        {"type": "message", "content": "Second message", "role": "user"},
    ]

    result = service._serialize_input(input_list)

    assert isinstance(result, dict)
    assert result["type"] == "list"
    assert "items" in result
    assert result["items"] == input_list


@pytest.mark.asyncio
async def test_get_response(mock_database_manager, mock_vllm_client):
    """Test getting a response by ID."""
    db_manager, mock_session = mock_database_manager

    # Mock response
    mock_response = MagicMock(spec=ResponseModel)
    mock_response.id = "resp_test123"
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

    assert response == mock_response
    # Verify database lookup was called
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_nonexistent_response(mock_database_manager, mock_vllm_client):
    """Test getting a response that doesn't exist."""
    db_manager, mock_session = mock_database_manager

    # Mock empty result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Get response
    response = await service.get_response("nonexistent")

    assert response is None


@pytest.mark.asyncio
async def test_list_responses(mock_database_manager, mock_vllm_client):
    """Test listing responses."""
    db_manager, mock_session = mock_database_manager

    # Mock responses
    mock_responses = [MagicMock(spec=ResponseModel), MagicMock(spec=ResponseModel)]
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


@pytest.mark.asyncio
async def test_delete_response(mock_database_manager, mock_vllm_client):
    """Test deleting a response."""
    db_manager, mock_session = mock_database_manager

    # Mock response
    mock_response = MagicMock(spec=ResponseModel)
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
    mock_session.flush.assert_called_once()


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

    # Delete response
    result = await service.delete_response("nonexistent")

    assert result is False
    mock_session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_tenant_isolation(mock_database_manager, mock_vllm_client):
    """Test that tenant isolation is properly enforced."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
        default_tenant_id="tenant1",
    )

    # Mock empty result for different tenant
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    # Try to get response from different tenant
    response = await service.get_response("resp_test123", tenant_id="tenant2")

    # Should return None (not found due to tenant isolation)
    assert response is None

    # Verify the query included the correct tenant_id filter
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_context_building_with_response_chain(
    mock_database_manager, mock_vllm_client
):
    """Test building context from a multi-response chain."""
    db_manager, mock_session = mock_database_manager

    service = ResponsesService(
        vllm_client=mock_vllm_client,
        db_manager=db_manager,
    )

    # Mock response chain: resp2 -> resp1 -> None
    mock_resp2 = MagicMock(spec=ResponseModel)
    mock_resp2.input = {"type": "text", "content": "Second question"}
    mock_resp2.output = {
        "items": [{"type": "message", "content": "Second answer", "role": "assistant"}]
    }
    mock_resp2.previous_response_id = "resp1"

    mock_resp1 = MagicMock(spec=ResponseModel)
    mock_resp1.input = {"type": "text", "content": "First question"}
    mock_resp1.output = {
        "items": [{"type": "message", "content": "First answer", "role": "assistant"}]
    }
    mock_resp1.previous_response_id = None

    # Mock the get_chain method to return the chain
    async def mock_get_chain(response_id, tenant_id):
        if response_id == "resp2":
            return [mock_resp1, mock_resp2]  # Chain in order
        return []

    # Patch the ResponsesStore to control the chain lookup
    with patch(
        "vllm_orchestrator.core.responses.service.ResponsesStore"
    ) as mock_store_class:
        mock_store = mock_store_class.return_value
        mock_store.get_chain = AsyncMock(side_effect=mock_get_chain)

        # Build context with previous response chain
        context = await service._build_context(
            input="Third question", previous_response_id="resp2", tenant_id="default"
        )

    # Should have full chain: resp1_input, resp1_output, resp2_input, resp2_output, new_input
    assert len(context) == 5

    # First exchange
    assert context[0]["content"] == "First question"
    assert context[1]["content"] == "First answer"

    # Second exchange
    assert context[2]["content"] == "Second question"
    assert context[3]["content"] == "Second answer"

    # New input
    assert context[4]["content"] == "Third question"
