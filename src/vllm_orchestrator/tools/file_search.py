"""File search tool executor using vector stores."""

import logging
from typing import Any, Dict, List, Optional

from vllm_orchestrator.core.vectors.service import VectorStoreService
from vllm_orchestrator.tools.base import ToolExecutor, ToolResult

logger = logging.getLogger(__name__)


class FileSearchExecutor(ToolExecutor):
    """Executor for file_search tool using vector stores."""

    def __init__(self, vector_service: VectorStoreService):
        super().__init__("file_search")
        self.vector_service = vector_service

    async def execute(
        self, tool_call_id: str, arguments: Dict[str, Any], **context: Any
    ) -> ToolResult:
        """Execute file search.

        Expected arguments:
        - query: str - Search query
        - vector_store_id: str - Vector store to search in
        - max_results: int - Maximum results (default 5)
        - score_threshold: float - Minimum similarity score (default 0.0)

        Args:
            tool_call_id: Unique identifier for this tool call
            arguments: Tool arguments
            **context: Additional context (tenant_id, etc.)

        Returns:
            ToolResult with search results
        """
        try:
            # Extract arguments
            query = arguments.get("query")
            vector_store_id = arguments.get("vector_store_id")
            max_results = arguments.get("max_results", 5)
            score_threshold = arguments.get("score_threshold", 0.0)
            tenant_id = context.get("tenant_id", "default")

            # Validate required arguments
            if not query:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    success=False,
                    content=None,
                    error="Missing required argument: query",
                )

            if not vector_store_id:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    success=False,
                    content=None,
                    error="Missing required argument: vector_store_id",
                )

            logger.info(
                f"Executing file search: query='{query}', "
                f"vector_store_id='{vector_store_id}', "
                f"max_results={max_results}, tenant_id='{tenant_id}'"
            )

            # Execute search
            search_results = await self.vector_service.search_vector_store(
                vector_store_id=vector_store_id,
                query=query,
                max_results=max_results,
                score_threshold=score_threshold,
                tenant_id=tenant_id,
            )

            # Format results for model consumption
            formatted_results = []
            for result in search_results:
                formatted_results.append(
                    {
                        "text": result.text,
                        "score": result.score,
                        "file_id": result.file_id,
                        "chunk_id": result.chunk_id,
                        "metadata": result.metadata or {},
                    }
                )

            content = {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
            }

            logger.info(
                f"File search completed: found {len(formatted_results)} results"
            )

            return ToolResult(
                tool_call_id=tool_call_id,
                success=True,
                content=content,
                metadata={
                    "vector_store_id": vector_store_id,
                    "max_results": max_results,
                    "score_threshold": score_threshold,
                },
            )

        except Exception as e:
            logger.error(f"File search failed: {e}", exc_info=True)
            return ToolResult(
                tool_call_id=tool_call_id,
                success=False,
                content=None,
                error=f"Search failed: {str(e)}",
            )

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get OpenAI tool definition for file search."""
        return {
            "type": "function",
            "function": {
                "name": "file_search",
                "description": (
                    "Search through uploaded files using semantic similarity. "
                    "This tool can find relevant document sections based on natural language queries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant document content",
                        },
                        "vector_store_id": {
                            "type": "string",
                            "description": "ID of the vector store to search in",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 5)",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Minimum similarity score for results (0.0-1.0, default 0.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.0,
                        },
                    },
                    "required": ["query", "vector_store_id"],
                    "additionalProperties": False,
                },
            },
        }


class FileSearchResult:
    """Individual search result from file search."""

    def __init__(
        self,
        text: str,
        score: float,
        file_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.score = score
        self.file_id = file_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
