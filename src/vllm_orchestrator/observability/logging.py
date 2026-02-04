"""Structured logging configuration for production."""

import logging
import logging.config
import sys
from typing import Any, Dict

import structlog


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    enable_access_logs: bool = True,
) -> None:
    """Configure structured logging with structlog.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output logs in JSON format
        enable_access_logs: Whether to enable HTTP access logs
    """
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.CallsiteParameterAdder(
            parameters=[structlog.processors.CallsiteParameter.FUNC_NAME]
        ),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=not json_logs)
                if not json_logs
                else structlog.processors.JSONRenderer(),
            },
        },
        "handlers": {
            "default": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            },
            "vllm_orchestrator": {
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO" if enable_access_logs else "WARNING",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"] if enable_access_logs else [],
                "level": "INFO" if enable_access_logs else "WARNING",
                "propagate": False,
            },
            "sqlalchemy": {
                "handlers": ["default"],
                "level": "WARNING",  # Reduce SQLAlchemy noise
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)

    # Get structured logger for application use
    logger = structlog.get_logger("vllm_orchestrator.logging")
    logger.info("Logging configured", level=level, json_logs=json_logs)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


# Request ID middleware for tracing
class RequestIDMiddleware:
    """Middleware to add request IDs to logs and responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            import uuid

            request_id = str(uuid.uuid4())

            # Add request ID to structlog context
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)

            # Add request ID to response headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = message.get("headers", [])
                    headers.append([b"x-request-id", request_id.encode()])
                    message["headers"] = headers
                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
