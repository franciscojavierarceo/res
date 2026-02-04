"""Prometheus metrics for monitoring and observability."""

import time
from typing import Any, Dict, List, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    Info,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response


class MetricsRegistry:
    """Central registry for application metrics."""

    def __init__(self):
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize Prometheus metrics."""

        # Application info
        self.app_info = Info(
            "vllm_orchestrator_app", "vLLM Orchestrator application information"
        )

        # HTTP request metrics
        self.http_requests_total = Counter(
            "vllm_orchestrator_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code", "tenant_id"],
        )

        self.http_request_duration_seconds = Histogram(
            "vllm_orchestrator_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint", "tenant_id"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # Response generation metrics
        self.responses_total = Counter(
            "vllm_orchestrator_responses_total",
            "Total responses generated",
            ["model", "status", "tenant_id", "has_tools"],
        )

        self.response_duration_seconds = Histogram(
            "vllm_orchestrator_response_duration_seconds",
            "Response generation duration in seconds",
            ["model", "tenant_id", "has_tools"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        self.response_tokens_total = Histogram(
            "vllm_orchestrator_response_tokens_total",
            "Total tokens in responses",
            ["model", "tenant_id", "token_type"],  # prompt, completion, total
            buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000],
        )

        # Tool execution metrics
        self.tool_calls_total = Counter(
            "vllm_orchestrator_tool_calls_total",
            "Total tool calls",
            ["tool_name", "status", "tenant_id"],  # status: success, failure
        )

        self.tool_execution_duration_seconds = Histogram(
            "vllm_orchestrator_tool_execution_duration_seconds",
            "Tool execution duration in seconds",
            ["tool_name", "tenant_id"],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Vector store metrics
        self.vector_searches_total = Counter(
            "vllm_orchestrator_vector_searches_total",
            "Total vector store searches",
            ["vector_store_id", "tenant_id"],
        )

        self.vector_search_duration_seconds = Histogram(
            "vllm_orchestrator_vector_search_duration_seconds",
            "Vector search duration in seconds",
            ["vector_store_id", "tenant_id"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        )

        self.vector_search_results_count = Histogram(
            "vllm_orchestrator_vector_search_results_count",
            "Number of results returned from vector searches",
            ["vector_store_id", "tenant_id"],
            buckets=[0, 1, 5, 10, 20, 50, 100],
        )

        # Database metrics
        self.database_operations_total = Counter(
            "vllm_orchestrator_database_operations_total",
            "Total database operations",
            [
                "operation",
                "table",
                "status",
            ],  # operation: select, insert, update, delete
        )

        self.database_operation_duration_seconds = Histogram(
            "vllm_orchestrator_database_operation_duration_seconds",
            "Database operation duration in seconds",
            ["operation", "table"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        )

        # vLLM client metrics
        self.vllm_requests_total = Counter(
            "vllm_orchestrator_vllm_requests_total",
            "Total requests to vLLM",
            [
                "model",
                "status",
                "request_type",
            ],  # request_type: generate, embed, tokenize
        )

        self.vllm_request_duration_seconds = Histogram(
            "vllm_orchestrator_vllm_request_duration_seconds",
            "vLLM request duration in seconds",
            ["model", "request_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
        )

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        tenant_id: str = "unknown",
    ) -> None:
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            tenant_id=tenant_id,
        ).inc()

        self.http_request_duration_seconds.labels(
            method=method, endpoint=endpoint, tenant_id=tenant_id
        ).observe(duration)

    def record_response_generation(
        self,
        model: str,
        status: str,
        duration: float,
        tenant_id: str,
        has_tools: bool = False,
        token_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record response generation metrics."""
        self.responses_total.labels(
            model=model,
            status=status,
            tenant_id=tenant_id,
            has_tools=str(has_tools).lower(),
        ).inc()

        self.response_duration_seconds.labels(
            model=model, tenant_id=tenant_id, has_tools=str(has_tools).lower()
        ).observe(duration)

        if token_counts:
            for token_type, count in token_counts.items():
                self.response_tokens_total.labels(
                    model=model, tenant_id=tenant_id, token_type=token_type
                ).observe(count)

    def record_tool_execution(
        self, tool_name: str, status: str, duration: float, tenant_id: str
    ) -> None:
        """Record tool execution metrics."""
        self.tool_calls_total.labels(
            tool_name=tool_name, status=status, tenant_id=tenant_id
        ).inc()

        self.tool_execution_duration_seconds.labels(
            tool_name=tool_name, tenant_id=tenant_id
        ).observe(duration)

    def record_vector_search(
        self, vector_store_id: str, duration: float, result_count: int, tenant_id: str
    ) -> None:
        """Record vector search metrics."""
        self.vector_searches_total.labels(
            vector_store_id=vector_store_id, tenant_id=tenant_id
        ).inc()

        self.vector_search_duration_seconds.labels(
            vector_store_id=vector_store_id, tenant_id=tenant_id
        ).observe(duration)

        self.vector_search_results_count.labels(
            vector_store_id=vector_store_id, tenant_id=tenant_id
        ).observe(result_count)

    def record_database_operation(
        self, operation: str, table: str, status: str, duration: float
    ) -> None:
        """Record database operation metrics."""
        self.database_operations_total.labels(
            operation=operation, table=table, status=status
        ).inc()

        self.database_operation_duration_seconds.labels(
            operation=operation, table=table
        ).observe(duration)

    def record_vllm_request(
        self, model: str, request_type: str, status: str, duration: float
    ) -> None:
        """Record vLLM request metrics."""
        self.vllm_requests_total.labels(
            model=model, status=status, request_type=request_type
        ).inc()

        self.vllm_request_duration_seconds.labels(
            model=model, request_type=request_type
        ).observe(duration)


# Global metrics registry
metrics_registry = MetricsRegistry()


def setup_metrics(app_name: str, version: str) -> None:
    """Set up application info metrics."""
    metrics_registry.app_info.info({"app_name": app_name, "version": version})


class MetricsMiddleware:
    """Middleware to automatically collect HTTP request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            status_code = 200

            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as e:
                status_code = 500
                raise
            finally:
                # Record metrics
                duration = time.time() - start_time
                method = scope.get("method", "UNKNOWN")
                path = scope.get("path", "/unknown")

                # Extract tenant_id if available from request
                tenant_id = "unknown"
                # In a real implementation, extract from headers/auth context

                metrics_registry.record_http_request(
                    method=method,
                    endpoint=self._normalize_path(path),
                    status_code=status_code,
                    duration=duration,
                    tenant_id=tenant_id,
                )
        else:
            await self.app(scope, receive, send)

    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics (remove IDs)."""
        import re

        # Replace UUIDs and IDs with placeholders
        path = re.sub(r"/[a-f0-9-]{36}", "/{uuid}", path)  # UUIDs
        path = re.sub(r"/\d+", "/{id}", path)  # Numeric IDs
        path = re.sub(r"/resp_[a-f0-9]+", "/resp_{id}", path)  # Response IDs
        path = re.sub(r"/vs_[a-f0-9]+", "/vs_{id}", path)  # Vector store IDs
        path = re.sub(r"/file_[a-f0-9]+", "/file_{id}", path)  # File IDs
        return path


def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
