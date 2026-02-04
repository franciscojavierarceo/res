"""FastAPI application entrypoint for vLLM Orchestrator."""

import logging
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vllm_orchestrator.api import responses_router, files_router, vector_stores_router
from vllm_orchestrator.config import Settings, StackConfig, settings
from vllm_orchestrator.core.responses import ResponsesService
from vllm_orchestrator.core.files.service import FilesService
from vllm_orchestrator.core.files.storage import LocalObjectStorage
from vllm_orchestrator.core.vectors.service import VectorStoreService
from vllm_orchestrator.tools.base import ToolRegistry
from vllm_orchestrator.tools.file_search import FileSearchExecutor
from vllm_orchestrator.tools.orchestrator import ToolOrchestrator
from vllm_orchestrator.storage.database import (
    close_database,
    init_database,
)
from vllm_orchestrator.vllm_client import VllmClient, create_vllm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(settings: Settings) -> StackConfig:
    """Load configuration from file or environment."""
    if settings.config_file and settings.config_file.exists():
        logger.info(f"Loading config from {settings.config_file}")
        with open(settings.config_file) as f:
            config_dict = yaml.safe_load(f)
        return StackConfig.from_dict(config_dict)
    else:
        logger.info("Using environment-based configuration")
        return settings.to_stack_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown of:
    - Database connections
    - vLLM client
    - Background services
    """
    logger.info("Starting vLLM Orchestrator...")

    # Load configuration
    config = load_config(settings)
    app.state.config = config

    # Initialize database
    logger.info("Initializing database...")
    db_manager = await init_database(config)
    app.state.db_manager = db_manager

    # Initialize vLLM client
    logger.info(f"Connecting to vLLM at {config.inference.base_url}...")
    vllm_client = create_vllm_client(config.inference)
    app.state.vllm_client = vllm_client

    # Check vLLM health
    if await vllm_client.health():
        logger.info("vLLM server is healthy")
    else:
        logger.warning("vLLM server health check failed - server may not be ready")

    # Initialize core services
    logger.info("Initializing core services...")

    # File storage service
    file_storage = LocalObjectStorage(base_path="/tmp/vllm_orchestrator_files")
    files_service = FilesService(object_storage=file_storage, db_manager=db_manager)
    app.state.files_service = files_service

    # Vector store service
    vector_store_service = VectorStoreService(
        db_manager=db_manager,
        vllm_client=vllm_client,
        files_service=files_service,
        default_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    app.state.vector_store_service = vector_store_service

    # Tool orchestration setup
    tool_registry = ToolRegistry()

    # Register file search tool
    file_search_executor = FileSearchExecutor(vector_service=vector_store_service)
    tool_registry.register(file_search_executor)

    # Create tool orchestrator
    tool_orchestrator = ToolOrchestrator(
        vllm_client=vllm_client,
        tool_registry=tool_registry
    )
    app.state.tool_orchestrator = tool_orchestrator

    # Initialize responses service with tool orchestration
    app.state.responses_service = ResponsesService(
        vllm_client=vllm_client,
        db_manager=db_manager,
        default_tenant_id=config.default_tenant_id,
        tool_orchestrator=tool_orchestrator,
    )

    logger.info("vLLM Orchestrator started successfully")

    yield

    # Shutdown
    logger.info("Shutting down vLLM Orchestrator...")
    await vllm_client.close()
    await close_database()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="vLLM Orchestrator",
        description="Stateful orchestration layer for vLLM with Responses API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure via settings in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(responses_router)
    app.include_router(files_router)
    app.include_router(vector_stores_router)

    # Health and utility endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        """List available models from vLLM."""
        vllm_client: VllmClient = request.app.state.vllm_client
        try:
            models = await vllm_client.list_models()
            return {"object": "list", "data": models}
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"error": f"Failed to fetch models: {str(e)}"},
            )

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "vLLM Orchestrator",
            "version": "0.1.0",
            "description": "Stateful orchestration layer for vLLM with Responses API and RAG capabilities",
            "endpoints": {
                "responses": "/v1/responses",
                "files": "/v1/files",
                "vector_stores": "/v1/vector_stores",
                "models": "/v1/models",
                "health": "/health",
                "docs": "/docs",
            },
        }

    return app


# Create the app instance
app = create_app()


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "vllm_orchestrator.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
