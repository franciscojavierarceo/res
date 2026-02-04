"""Tool orchestration for multi-turn inference loops."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai.types.responses import Response
from openai.types.responses.response_input_item_param import ResponseInputItemParam

from vllm_orchestrator.tools.base import ToolRegistry, ToolExecutionError
from vllm_orchestrator.vllm_client import VllmClient

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """Orchestrates tool execution during response generation.

    This class handles the multi-turn inference loop:
    1. Send prompt to vLLM
    2. Detect tool calls in response
    3. Execute tools
    4. Append tool results to context
    5. Continue generation with updated context
    """

    def __init__(self, vllm_client: VllmClient, tool_registry: ToolRegistry):
        self.vllm_client = vllm_client
        self.tool_registry = tool_registry
        self.max_tool_iterations = 10  # Prevent infinite loops

    async def create_response_with_tools(
        self,
        model: str,
        input: List[ResponseInputItemParam],
        *,
        instructions: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: Dict[str, str] | None = None,
        stream: bool = False,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> Union[Response, AsyncIterator[Any]]:
        """Create response with tool execution support.

        Args:
            model: Model identifier
            input: Input messages/items
            instructions: System instructions
            tools: Tool definitions (if None, uses all registered tools)
            tool_choice: Tool choice strategy
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            metadata: Request metadata
            stream: Whether to stream the response
            tenant_id: Tenant identifier
            **kwargs: Additional parameters

        Returns:
            Response object or async iterator of streaming events
        """
        # Use all registered tools if none specified
        if tools is None:
            tools = self.tool_registry.get_tool_definitions()

        # Build initial context
        context = list(input)
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            logger.info(f"Tool orchestration iteration {iteration}")

            # Generate response from vLLM
            vllm_response = await self.vllm_client.create_response(
                model=model,
                input=context,
                instructions=instructions,
                tools=tools,
                tool_choice=tool_choice,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                metadata=metadata,
                stream=False,  # Always non-streaming for tool execution
                **kwargs,
            )

            assert isinstance(vllm_response, Response)

            # Check if response contains tool calls
            tool_calls = self._extract_tool_calls(vllm_response)

            if not tool_calls:
                # No tool calls - we're done
                logger.info("No tool calls found, orchestration complete")
                if stream:
                    return self._convert_to_stream(vllm_response)
                else:
                    return vllm_response

            # Add the assistant's response with tool calls to context
            context.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": self._extract_content(vllm_response),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute tool calls
            tool_results = await self._execute_tool_calls(
                tool_calls, tenant_id=tenant_id
            )

            # Add tool results to context
            for result in tool_results:
                context.append(
                    {
                        "type": "message",
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": self._format_tool_result(result),
                    }
                )

            # Continue loop to generate next response with tool results

        # If we hit max iterations, return the last response
        logger.warning(f"Hit maximum tool iterations ({self.max_tool_iterations})")
        if stream:
            return self._convert_to_stream(vllm_response)
        else:
            return vllm_response

    def _extract_tool_calls(self, response: Response) -> List[Dict[str, Any]]:
        """Extract tool calls from vLLM response.

        Args:
            response: vLLM response object

        Returns:
            List of tool call dicts with id, name, arguments
        """
        tool_calls = []

        # Check output items for tool calls
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type") and item.type == "function_call":
                    tool_calls.append(
                        {
                            "id": getattr(item, "id", f"call_{len(tool_calls)}"),
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

        return tool_calls

    def _extract_content(self, response: Response) -> str:
        """Extract text content from response."""
        if hasattr(response, "output") and response.output:
            content_parts = []
            for item in response.output:
                if hasattr(item, "type") and item.type == "text":
                    content_parts.append(item.text)
            return "".join(content_parts)
        return ""

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], **context: Any
    ) -> List[Any]:
        """Execute multiple tool calls concurrently.

        Args:
            tool_calls: List of tool call dicts
            **context: Additional context for tool execution

        Returns:
            List of ToolResult objects
        """
        tasks = []
        for tool_call in tool_calls:
            task = self._execute_single_tool_call(tool_call, **context)
            tasks.append(task)

        if tasks:
            logger.info(f"Executing {len(tasks)} tool calls concurrently")
            return await asyncio.gather(*tasks)
        else:
            return []

    async def _execute_single_tool_call(
        self, tool_call: Dict[str, Any], **context: Any
    ) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dict with id, name, arguments
            **context: Additional context

        Returns:
            ToolResult object
        """
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        arguments = tool_call.get("arguments", {})

        logger.info(f"Executing tool call: {tool_name} (id={tool_call_id})")

        try:
            return await self.tool_registry.execute_tool(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                **context,
            )
        except ToolExecutionError as e:
            logger.error(f"Tool execution failed: {e}")
            from vllm_orchestrator.tools.base import ToolResult

            return ToolResult(
                tool_call_id=tool_call_id, success=False, content=None, error=str(e)
            )

    def _format_tool_result(self, result: Any) -> str:
        """Format tool result for model consumption.

        Args:
            result: ToolResult object

        Returns:
            Formatted string for the model
        """
        if result.success:
            if isinstance(result.content, dict):
                # Format structured content
                if "results" in result.content:
                    # File search results
                    query = result.content.get("query", "")
                    results = result.content.get("results", [])
                    total = result.content.get("total_results", len(results))

                    if not results:
                        return f"No results found for query: '{query}'"

                    formatted = f"Found {total} results for query '{query}':\n\n"
                    for i, res in enumerate(results, 1):
                        text = res.get("text", "")
                        score = res.get("score", 0.0)
                        file_id = res.get("file_id", "unknown")

                        formatted += (
                            f"{i}. [Score: {score:.3f}] [File: {file_id}]\n{text}\n\n"
                        )

                    return formatted.strip()
                else:
                    # Generic structured content
                    return str(result.content)
            else:
                return str(result.content)
        else:
            error_msg = result.error or "Unknown error"
            return f"Tool execution failed: {error_msg}"

    async def _convert_to_stream(self, response: Response) -> AsyncIterator[Any]:
        """Convert a non-streaming response to streaming format.

        This is used when the final response needs to be streamed
        but tool execution requires non-streaming mode.
        """
        # Emit response events in streaming format
        yield {"type": "response.started", "response": response.model_dump()}

        if hasattr(response, "output") and response.output:
            for item in response.output:
                yield {
                    "type": "response.output_item.added",
                    "item": item.model_dump() if hasattr(item, "model_dump") else item,
                }

                if hasattr(item, "type") and item.type == "text":
                    # Emit text deltas if needed
                    yield {"type": "response.text.delta", "delta": item.text}

                yield {
                    "type": "response.output_item.done",
                    "item": item.model_dump() if hasattr(item, "model_dump") else item,
                }

        yield {"type": "response.done", "response": response.model_dump()}
