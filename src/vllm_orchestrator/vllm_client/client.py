"""HTTP client for communicating with vLLM inference server.

Uses the OpenAI SDK since vLLM exposes OpenAI-compatible endpoints.
The orchestrator always calls vLLM with store=False to keep vLLM stateless.
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.responses import Response
from openai.types.responses.response_input_item_param import ResponseInputItemParam

from vllm_orchestrator.config import (
    InferenceBackendConfig,
    OpenAIInferenceConfig,
    VllmInferenceConfig,
)


class VllmClient:
    """Client for vLLM inference server.

    Wraps the OpenAI SDK to communicate with vLLM's OpenAI-compatible API.
    All requests are made with store=False since the orchestrator handles storage.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "EMPTY"  # vLLM accepts any key if not configured
        self._timeout = timeout

        # OpenAI SDK client for type-safe API calls
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=timeout,
            http_client=httpx.AsyncClient(timeout=timeout),
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

    async def health(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from vLLM."""
        models = await self._client.models.list()
        return [model.model_dump() for model in models.data]

    async def create_response(
        self,
        model: str,
        input: str | list[ResponseInputItemParam],
        *,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Response | AsyncIterator[Any]:
        """Create a response using vLLM's Responses API.

        Always sets store=False since the orchestrator handles persistence.

        Args:
            model: Model identifier
            input: User input (string or list of input items)
            instructions: System instructions
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: Tool definitions
            tool_choice: Tool choice strategy
            metadata: Request metadata
            stream: Whether to stream the response
            **kwargs: Additional parameters passed to vLLM

        Returns:
            Response object or async iterator of streaming events
        """
        # Build request parameters
        params: dict[str, Any] = {
            "model": model,
            "input": input,
            "store": False,  # Always stateless - orchestrator handles storage
        }

        if instructions is not None:
            params["instructions"] = instructions
        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if metadata is not None:
            params["metadata"] = metadata

        # Add any extra kwargs (vLLM-specific parameters)
        params.update(kwargs)

        if stream:
            return self._stream_response(params)
        else:
            return await self._client.responses.create(**params)

    async def _stream_response(self, params: dict[str, Any]) -> AsyncIterator[Any]:
        """Stream response events from vLLM."""
        params["stream"] = True
        async with self._client.responses.create(**params) as stream:
            async for event in stream:
                yield event

    async def create_embeddings(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        encoding_format: str = "float",
    ) -> dict[str, Any]:
        """Generate embeddings using vLLM's embeddings endpoint.

        Args:
            model: Embedding model identifier
            input: Text or list of texts to embed
            dimensions: Output embedding dimensions (if supported)
            encoding_format: Output format ("float" or "base64")

        Returns:
            Embeddings response with vectors
        """
        params: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            params["dimensions"] = dimensions

        response = await self._client.embeddings.create(**params)
        return response.model_dump()


def create_vllm_client(config: InferenceBackendConfig) -> VllmClient:
    """Factory function to create VllmClient from config.

    Args:
        config: Inference backend configuration (VllmInferenceConfig or OpenAIInferenceConfig)

    Returns:
        Configured VllmClient instance
    """
    match config:
        case VllmInferenceConfig():
            return VllmClient(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        case OpenAIInferenceConfig():
            return VllmClient(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        case _:
            raise ValueError(f"Unknown inference backend type: {type(config)}")
