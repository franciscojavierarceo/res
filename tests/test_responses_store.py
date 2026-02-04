"""Tests for responses store."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_orchestrator.core.responses.store import ResponsesStore
from vllm_orchestrator.storage.models import ResponseModel


@pytest.fixture
async def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.delete = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_create_response(mock_session):
    """Test creating a response record."""
    store = ResponsesStore(mock_session)

    input_data = {"type": "message", "content": "Hello world"}

    # Create response
    response = await store.create(
        response_id="resp_test123",
        model="test-model",
        input_data=input_data,
        tenant_id="default",
        status="in_progress",
        metadata={"source": "api"},
    )

    # Verify response model was created with correct data
    assert isinstance(response, ResponseModel)
    assert response.id == "resp_test123"
    assert response.model == "test-model"
    assert response.input == input_data
    assert response.status == "in_progress"
    assert response.tenant_id == "default"
    assert response.metadata_ == {"source": "api"}

    # Verify session operations
    mock_session.add.assert_called_once_with(response)
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_get_response(mock_session):
    """Test getting a response by ID."""
    store = ResponsesStore(mock_session)

    # Mock response lookup
    mock_response = MagicMock(spec=ResponseModel)
    mock_response.id = "resp_test123"
    mock_response.status = "completed"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_response
    mock_session.execute.return_value = mock_result

    # Get response
    response = await store.get("resp_test123", "default")

    assert response == mock_response
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_nonexistent_response(mock_session):
    """Test getting a response that doesn't exist."""
    store = ResponsesStore(mock_session)

    # Mock empty result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    # Get response
    response = await store.get("nonexistent", "default")

    assert response is None


@pytest.mark.asyncio
async def test_update_response(mock_session):
    """Test updating a response record."""
    store = ResponsesStore(mock_session)

    # Mock existing response
    mock_response = MagicMock(spec=ResponseModel)
    mock_response.status = "in_progress"

    # Set up the store.get() to return our mock
    async def mock_get(response_id, tenant_id):
        return mock_response

    store.get = AsyncMock(side_effect=mock_get)

    # Update response
    updated_response = await store.update(
        response_id="resp_test123",
        tenant_id="default",
        status="completed",
        output={"items": [{"type": "message", "content": "Done"}]},
        usage={"total_tokens": 50},
    )

    # Verify updates
    assert mock_response.status == "completed"
    assert mock_response.output == {"items": [{"type": "message", "content": "Done"}]}
    assert mock_response.usage == {"total_tokens": 50}
    assert updated_response == mock_response

    # Verify session flush
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_update_nonexistent_response(mock_session):
    """Test updating a response that doesn't exist."""
    store = ResponsesStore(mock_session)

    # Set up the store.get() to return None
    store.get = AsyncMock(return_value=None)

    # Try to update nonexistent response
    result = await store.update(
        response_id="nonexistent",
        tenant_id="default",
        status="completed",
    )

    assert result is None
    mock_session.flush.assert_not_called()


@pytest.mark.asyncio
async def test_get_chain(mock_session):
    """Test getting a response chain."""
    store = ResponsesStore(mock_session)

    # Mock response chain: resp3 -> resp2 -> resp1 -> None
    mock_resp3 = MagicMock(spec=ResponseModel)
    mock_resp3.id = "resp_3"
    mock_resp3.previous_response_id = "resp_2"

    mock_resp2 = MagicMock(spec=ResponseModel)
    mock_resp2.id = "resp_2"
    mock_resp2.previous_response_id = "resp_1"

    mock_resp1 = MagicMock(spec=ResponseModel)
    mock_resp1.id = "resp_1"
    mock_resp1.previous_response_id = None

    # Mock get to return appropriate responses
    async def mock_get(response_id, tenant_id):
        if response_id == "resp_3":
            return mock_resp3
        elif response_id == "resp_2":
            return mock_resp2
        elif response_id == "resp_1":
            return mock_resp1
        return None

    store.get = AsyncMock(side_effect=mock_get)

    # Get chain starting from resp_3
    chain = await store.get_chain("resp_3", "default")

    # Should return chain in order: [resp_1, resp_2, resp_3]
    assert len(chain) == 3
    assert chain[0] == mock_resp1
    assert chain[1] == mock_resp2
    assert chain[2] == mock_resp3


@pytest.mark.asyncio
async def test_get_chain_single_response(mock_session):
    """Test getting a chain for a single response (no previous)."""
    store = ResponsesStore(mock_session)

    # Mock single response with no previous
    mock_response = MagicMock(spec=ResponseModel)
    mock_response.id = "resp_single"
    mock_response.previous_response_id = None

    store.get = AsyncMock(return_value=mock_response)

    # Get chain
    chain = await store.get_chain("resp_single", "default")

    # Should return just the single response
    assert len(chain) == 1
    assert chain[0] == mock_response


@pytest.mark.asyncio
async def test_get_chain_nonexistent(mock_session):
    """Test getting a chain for a nonexistent response."""
    store = ResponsesStore(mock_session)

    store.get = AsyncMock(return_value=None)

    # Get chain for nonexistent response
    chain = await store.get_chain("nonexistent", "default")

    # Should return empty chain
    assert chain == []


@pytest.mark.asyncio
async def test_list_responses(mock_session):
    """Test listing responses."""
    store = ResponsesStore(mock_session)

    # Mock response list
    mock_responses = [MagicMock(spec=ResponseModel), MagicMock(spec=ResponseModel)]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_responses
    mock_session.execute.return_value = mock_result

    # List responses
    responses = await store.list_responses(tenant_id="default", limit=10)

    assert responses == mock_responses
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_response(mock_session):
    """Test deleting a response."""
    store = ResponsesStore(mock_session)

    # Mock existing response
    mock_response = MagicMock(spec=ResponseModel)
    store.get = AsyncMock(return_value=mock_response)

    # Delete response
    result = await store.delete("resp_test123", "default")

    assert result is True
    mock_session.delete.assert_called_once_with(mock_response)
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_delete_nonexistent_response(mock_session):
    """Test deleting a nonexistent response."""
    store = ResponsesStore(mock_session)

    # Mock nonexistent response
    store.get = AsyncMock(return_value=None)

    # Try to delete nonexistent response
    result = await store.delete("nonexistent", "default")

    assert result is False
    mock_session.delete.assert_not_called()
    mock_session.flush.assert_not_called()


@pytest.mark.asyncio
async def test_response_with_previous_id(mock_session):
    """Test creating a response with previous_response_id."""
    store = ResponsesStore(mock_session)

    # Create response with previous ID
    response = await store.create(
        response_id="resp_follow_up",
        model="test-model",
        input_data={"type": "message", "content": "Follow up question"},
        tenant_id="default",
        previous_response_id="resp_previous",
    )

    assert response.previous_response_id == "resp_previous"


@pytest.mark.asyncio
async def test_multi_tenant_isolation(mock_session):
    """Test that responses are properly isolated by tenant."""
    store = ResponsesStore(mock_session)

    # Create responses for different tenants
    tenant1_response = await store.create(
        response_id="resp_tenant1",
        model="test-model",
        input_data={"type": "message", "content": "Tenant 1 message"},
        tenant_id="tenant1",
    )

    tenant2_response = await store.create(
        response_id="resp_tenant2",
        model="test-model",
        input_data={"type": "message", "content": "Tenant 2 message"},
        tenant_id="tenant2",
    )

    assert tenant1_response.tenant_id == "tenant1"
    assert tenant2_response.tenant_id == "tenant2"

    # Verify both were added to session
    assert mock_session.add.call_count == 2
