"""Base classes for tool execution."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from tool execution."""

    tool_call_id: str
    success: bool
    content: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolExecutor(ABC):
    """Abstract base class for tool executors.

    Each tool type (file_search, web_search, code_interpreter)
    should implement this interface.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(
        self, tool_call_id: str, arguments: Dict[str, Any], **context: Any
    ) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_call_id: Unique identifier for this tool call
            arguments: Tool arguments from the model
            **context: Additional context (tenant_id, user_id, etc.)

        Returns:
            ToolResult with execution outcome
        """
        pass

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get OpenAI tool definition for this executor.

        Returns:
            Tool definition dict compatible with OpenAI API
        """
        pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, error_message: str):
        self.tool_name = tool_name
        self.error_message = error_message
        super().__init__(f"Tool '{tool_name}' failed: {error_message}")


class ToolRegistry:
    """Registry for available tool executors."""

    def __init__(self):
        self._executors: Dict[str, ToolExecutor] = {}

    def register(self, executor: ToolExecutor) -> None:
        """Register a tool executor."""
        self._executors[executor.name] = executor

    def get_executor(self, tool_name: str) -> Optional[ToolExecutor]:
        """Get executor for a tool name."""
        return self._executors.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._executors.keys())

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI tool definitions for all registered tools."""
        return [executor.get_tool_definition() for executor in self._executors.values()]

    async def execute_tool(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
        **context: Any,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_call_id: Unique identifier for this tool call
            arguments: Tool arguments
            **context: Additional context

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolExecutionError: If tool not found or execution fails
        """
        executor = self.get_executor(tool_name)
        if executor is None:
            raise ToolExecutionError(tool_name, f"Tool '{tool_name}' not registered")

        try:
            return await executor.execute(tool_call_id, arguments, **context)
        except Exception as e:
            raise ToolExecutionError(tool_name, str(e)) from e
