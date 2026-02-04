"""vLLM client for inference communication."""

from vllm_orchestrator.vllm_client.client import VllmClient, create_vllm_client

__all__ = [
    "VllmClient",
    "create_vllm_client",
]
