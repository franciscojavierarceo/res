"""Tests for tool orchestration framework."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_orchestrator.core.vectors.service import VectorStoreService
from vllm_orchestrator.tools.base import ToolExecutor, ToolRegistry, ToolResult
from vllm_orchestrator.tools.file_search import FileSearchExecutor, FileSearchResult
from vllm_orchestrator.tools.orchestrator import ToolOrchestrator
from vllm_orchestrator.vllm_client import VllmClient


class MockToolExecutor(ToolExecutor):
    """Mock tool executor for testing."""

    def __init__(self, name: str = "mock_tool"):
        super().__init__(name)
        self.execute_calls = []

    async def execute(
        self, tool_call_id: str, arguments: dict, **context
    ) -> ToolResult:
        """Mock tool execution."""
        self.execute_calls.append(
            {"tool_call_id": tool_call_id, "arguments": arguments, "context": context}
        )

        # Return mock result
        return ToolResult(
            tool_call_id=tool_call_id,
            success=True,
            content=f"Mock result for {self.name}",
            metadata={"test": True},
        )

    def get_tool_definition(self) -> dict:
        """Mock tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"Mock {self.name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Test query"}
                    },
                    "required": ["query"],
                },
            },
        }


@pytest.fixture
def mock_vector_service():
    """Mock vector store service."""
    service = MagicMock(spec=VectorStoreService)

    # Mock search results
    async def mock_search(
        vector_store_id, query, max_results=5, score_threshold=0.0, tenant_id="default"
    ):
        if "empty" in query.lower():
            return []

        return [
            MagicMock(
                text=f"Sample text result for '{query}'",
                score=0.85,
                file_id="file_123",
                chunk_id="chunk_456",
                metadata={"source": "test"},
            ),
            MagicMock(
                text=f"Another result for '{query}'",
                score=0.72,
                file_id="file_789",
                chunk_id="chunk_012",
                metadata={"source": "test2"},
            ),
        ]

    service.search_vector_store = AsyncMock(side_effect=mock_search)
    return service


@pytest.fixture
def mock_vllm_client():
    """Mock vLLM client."""
    client = MagicMock(spec=VllmClient)

    # Mock response creation
    async def mock_create_response(**kwargs):
        from openai.types.responses import Response

        # Create a mock response with tool calls
        response = MagicMock(spec=Response)
        response.id = "resp_mock"
        response.model = kwargs.get("model", "test-model")

        # Check if this is a tool execution iteration
        input_messages = kwargs.get("input", [])
        has_tool_results = any(
            msg.get("role") == "tool" for msg in input_messages if isinstance(msg, dict)
        )

        if has_tool_results:
            # This is after tool execution - return final response
            mock_text_item = MagicMock()
            mock_text_item.type = "text"
            mock_text_item.text = (
                "Based on the search results, I found relevant information."
            )
            response.output = [mock_text_item]
        else:
            # First call - return response with tool call
            mock_tool_call = MagicMock()
            mock_tool_call.type = "function_call"
            mock_tool_call.id = "call_123"
            mock_tool_call.name = "file_search"
            mock_tool_call.arguments = {
                "query": "test query",
                "vector_store_id": "vs_123",
            }
            response.output = [mock_tool_call]

        response.model_dump.return_value = {
            "id": response.id,
            "model": response.model,
            "output": [
                {
                    "type": item.type,
                    "text": getattr(item, "text", ""),
                    "id": getattr(item, "id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", {}),
                }
                for item in response.output
            ],
        }

        return response

    client.create_response = AsyncMock(side_effect=mock_create_response)
    return client


def test_tool_registry():
    """Test tool registry functionality."""
    registry = ToolRegistry()

    # Test empty registry
    assert registry.list_tools() == []
    assert registry.get_tool_definitions() == []
    assert registry.get_executor("nonexistent") is None

    # Register a tool
    mock_tool = MockToolExecutor("test_tool")
    registry.register(mock_tool)

    # Test registry with tool
    assert registry.list_tools() == ["test_tool"]
    assert len(registry.get_tool_definitions()) == 1
    assert registry.get_executor("test_tool") == mock_tool
    assert registry.get_executor("nonexistent") is None


@pytest.mark.asyncio
async def test_tool_registry_execute():
    """Test tool execution through registry."""
    registry = ToolRegistry()
    mock_tool = MockToolExecutor("test_tool")
    registry.register(mock_tool)

    # Execute tool
    result = await registry.execute_tool(
        tool_name="test_tool",
        tool_call_id="call_123",
        arguments={"query": "test"},
        tenant_id="test_tenant",
    )

    # Verify result
    assert result.tool_call_id == "call_123"
    assert result.success is True
    assert result.content == "Mock result for test_tool"

    # Verify tool was called correctly
    assert len(mock_tool.execute_calls) == 1
    call = mock_tool.execute_calls[0]
    assert call["tool_call_id"] == "call_123"
    assert call["arguments"] == {"query": "test"}
    assert call["context"]["tenant_id"] == "test_tenant"


@pytest.mark.asyncio
async def test_tool_registry_execute_nonexistent():
    """Test executing nonexistent tool."""
    registry = ToolRegistry()

    with pytest.raises(Exception) as exc_info:
        await registry.execute_tool(
            tool_name="nonexistent", tool_call_id="call_123", arguments={}
        )

    assert "Tool 'nonexistent' not registered" in str(exc_info.value)


@pytest.mark.asyncio
async def test_file_search_executor_success(mock_vector_service):
    """Test successful file search execution."""
    executor = FileSearchExecutor(mock_vector_service)

    # Test tool definition
    tool_def = executor.get_tool_definition()
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "file_search"
    assert "query" in tool_def["function"]["parameters"]["properties"]

    # Execute search
    result = await executor.execute(
        tool_call_id="call_123",
        arguments={
            "query": "test query",
            "vector_store_id": "vs_123",
            "max_results": 3,
        },
        tenant_id="test_tenant",
    )

    # Verify result
    assert result.tool_call_id == "call_123"
    assert result.success is True
    assert isinstance(result.content, dict)
    assert result.content["query"] == "test query"
    assert result.content["total_results"] == 2
    assert len(result.content["results"]) == 2

    # Verify vector service was called
    mock_vector_service.search_vector_store.assert_called_once_with(
        vector_store_id="vs_123",
        query="test query",
        max_results=3,
        score_threshold=0.0,
        tenant_id="test_tenant",
    )


@pytest.mark.asyncio
async def test_file_search_executor_missing_args(mock_vector_service):
    """Test file search with missing arguments."""
    executor = FileSearchExecutor(mock_vector_service)

    # Missing query
    result = await executor.execute(
        tool_call_id="call_123",
        arguments={"vector_store_id": "vs_123"},
        tenant_id="test_tenant",
    )

    assert result.success is False
    assert "Missing required argument: query" in result.error

    # Missing vector_store_id
    result = await executor.execute(
        tool_call_id="call_456", arguments={"query": "test"}, tenant_id="test_tenant"
    )

    assert result.success is False
    assert "Missing required argument: vector_store_id" in result.error


@pytest.mark.asyncio
async def test_file_search_executor_empty_results(mock_vector_service):
    """Test file search with no results."""
    executor = FileSearchExecutor(mock_vector_service)

    result = await executor.execute(
        tool_call_id="call_123",
        arguments={
            "query": "empty query",  # This triggers empty results in mock
            "vector_store_id": "vs_123",
        },
        tenant_id="test_tenant",
    )

    assert result.success is True
    assert result.content["total_results"] == 0
    assert len(result.content["results"]) == 0


@pytest.mark.asyncio
async def test_tool_orchestrator_no_tools(mock_vllm_client):
    """Test orchestrator with no tool calls in response."""
    registry = ToolRegistry()
    orchestrator = ToolOrchestrator(mock_vllm_client, registry)

    # Mock response with no tool calls
    async def mock_no_tools(**kwargs):
        from openai.types.responses import Response

        response = MagicMock(spec=Response)
        response.output = [MagicMock(type="text", text="Simple response")]
        response.model_dump.return_value = {
            "output": [{"type": "text", "text": "Simple response"}]
        }
        return response

    mock_vllm_client.create_response = AsyncMock(side_effect=mock_no_tools)

    # Execute
    response = await orchestrator.create_response_with_tools(
        model="test-model",
        input=[{"type": "message", "role": "user", "content": "Hello"}],
        tenant_id="test_tenant",
    )

    # Should return response directly
    assert response is not None
    mock_vllm_client.create_response.assert_called_once()


@pytest.mark.asyncio
async def test_tool_orchestrator_with_tools(mock_vllm_client, mock_vector_service):
    """Test orchestrator with tool calls."""
    registry = ToolRegistry()
    file_search = FileSearchExecutor(mock_vector_service)
    registry.register(file_search)

    orchestrator = ToolOrchestrator(mock_vllm_client, registry)

    # Execute with tools
    response = await orchestrator.create_response_with_tools(
        model="test-model",
        input=[
            {"type": "message", "role": "user", "content": "Search for information"}
        ],
        tools=[file_search.get_tool_definition()],
        tenant_id="test_tenant",
    )

    # Should have made multiple calls to vLLM (tool call + final response)
    assert mock_vllm_client.create_response.call_count == 2

    # Verify vector search was called
    mock_vector_service.search_vector_store.assert_called_once()


@pytest.mark.asyncio
async def test_tool_orchestrator_max_iterations(mock_vllm_client):
    """Test orchestrator with max iteration limit."""
    registry = ToolRegistry()
    mock_tool = MockToolExecutor("infinite_tool")
    registry.register(mock_tool)

    orchestrator = ToolOrchestrator(mock_vllm_client, registry)
    orchestrator.max_tool_iterations = 2  # Set low limit for testing

    # Mock vLLM to always return tool calls
    async def mock_infinite_tools(**kwargs):
        from openai.types.responses import Response

        response = MagicMock(spec=Response)

        mock_tool_call = MagicMock()
        mock_tool_call.type = "function_call"
        mock_tool_call.id = "call_infinite"
        mock_tool_call.name = "infinite_tool"
        mock_tool_call.arguments = {"query": "infinite"}

        response.output = [mock_tool_call]
        response.model_dump.return_value = {"output": [{"type": "function_call"}]}
        return response

    mock_vllm_client.create_response = AsyncMock(side_effect=mock_infinite_tools)

    # Execute
    response = await orchestrator.create_response_with_tools(
        model="test-model",
        input=[{"type": "message", "role": "user", "content": "Start infinite loop"}],
        tools=[mock_tool.get_tool_definition()],
        tenant_id="test_tenant",
    )

    # Should hit max iterations
    assert mock_vllm_client.create_response.call_count == 2  # max_tool_iterations
    assert len(mock_tool.execute_calls) == 2
