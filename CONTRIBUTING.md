# Contributing to vLLM Orchestrator

We welcome contributions to vLLM Orchestrator! This guide will help you set up your development environment, understand our coding standards, and contribute effectively.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [Contributing Workflow](#contributing-workflow)
- [Architecture Guidelines](#architecture-guidelines)
- [Database Changes](#database-changes)
- [Adding New Tools](#adding-new-tools)
- [Debugging](#debugging)

## Development Environment Setup

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **PostgreSQL 15+** with pgvector extension (or SQLite for quick development)
- **Redis 7+** (optional but recommended for full feature testing)
- **Docker and Docker Compose** (for integration testing)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-orchestrator.git
cd vllm-orchestrator

# Install dependencies
pip install -e ".[dev]"

# Or with uv (recommended for faster installs)
uv sync --extra dev

# Start development services
docker-compose up -d postgres redis

# Set environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vllm_orchestrator"
export REDIS_URL="redis://localhost:6379"
export VLLM_BASE_URL="http://localhost:8001"  # Point to your vLLM instance

# Run database migrations
python -m vllm_orchestrator.storage.migrations

# Verify setup
python -c "from vllm_orchestrator.storage.database import create_engine; print('✅ Database connected')"
```

### Development with SQLite (No External Dependencies)

```bash
# Use SQLite for quick development (no Docker required)
export DATABASE_URL="sqlite+aiosqlite:///./dev.db"
export VLLM_BASE_URL="http://localhost:8001"

# Skip Redis (disables caching but everything else works)
unset REDIS_URL

python -m vllm_orchestrator.main
```

### Running a Development vLLM Server

```bash
# Install vLLM
pip install vllm

# Start vLLM server with a small model for testing
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --port 8001 \
    --max-model-len 2048

# Or use a larger model if you have GPU
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --port 8001
```

## Project Structure

```
vllm-orchestrator/
├── src/vllm_orchestrator/       # Main package
│   ├── api/                     # FastAPI routers
│   │   ├── responses.py         # Responses API endpoints
│   │   ├── files.py             # Files API endpoints
│   │   └── vector_stores.py     # Vector stores API endpoints
│   │
│   ├── core/                    # Core business logic
│   │   ├── responses/           # Response generation and storage
│   │   ├── files/               # File storage and processing
│   │   └── vectors/             # Vector store and search
│   │
│   ├── tools/                   # Tool execution framework
│   │   ├── base.py              # Abstract base classes
│   │   ├── file_search.py       # File search tool implementation
│   │   ├── registry.py          # Tool registry
│   │   └── orchestrator.py      # Multi-turn tool orchestration
│   │
│   ├── vllm_client/             # vLLM communication layer
│   │   ├── client.py            # HTTP client for vLLM
│   │   └── protocol.py          # Request/response types
│   │
│   ├── storage/                 # Database layer
│   │   ├── database.py          # SQLAlchemy setup
│   │   ├── models.py            # ORM models
│   │   └── migrations.py        # Database migrations
│   │
│   ├── observability/           # Production monitoring
│   │   ├── logging.py           # Structured logging
│   │   └── metrics.py           # Prometheus metrics
│   │
│   ├── middleware/              # HTTP middleware
│   │   ├── security.py          # Security headers, CORS
│   │   └── rate_limiting.py     # Redis-based rate limiting
│   │
│   ├── performance/             # Performance optimization
│   │   ├── connection_pool.py   # Database connection pooling
│   │   └── caching.py           # Redis caching layer
│   │
│   ├── config.py                # Configuration management
│   ├── main.py                  # FastAPI application
│   └── main_production.py       # Production-hardened app
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── load_test.py             # Load testing utility
│   └── conftest.py              # pytest configuration
│
├── deployment/                  # Deployment configuration
│   ├── Dockerfile               # Production container
│   ├── docker-compose.yml       # Development setup
│   ├── docker-compose.production.yml
│   ├── kubernetes/              # K8s manifests
│   └── prometheus.yml           # Monitoring config
│
└── examples/                    # Example applications
    ├── basic_usage.py
    ├── file_search_example.py
    └── streaming_example.py
```

## Running Tests

### Full Test Suite

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=vllm_orchestrator --cov-report=html

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest -k "test_responses"            # Tests matching pattern
pytest tests/test_responses.py -v     # Verbose output for specific file
```

### Test Categories

1. **Unit Tests** (`tests/unit/`): Fast tests that don't require external services
2. **Integration Tests** (`tests/integration/`): Tests that require database/Redis
3. **Load Tests** (`tests/load_test.py`): Performance and stress testing

### Test Database

Tests use a separate test database. Set up with:

```bash
# Create test database
createdb vllm_orchestrator_test

# Set test environment
export TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vllm_orchestrator_test"

# Tests will automatically create/drop tables
pytest
```

### Running Tests with Docker

```bash
# Run tests in containerized environment
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Or build a test image
docker build -t vllm-orchestrator-test -f Dockerfile.test .
docker run --rm vllm-orchestrator-test
```

## Code Quality

### Linting and Formatting

We use **Ruff** for both linting and formatting:

```bash
# Check code style
ruff check src tests

# Fix auto-fixable issues
ruff check --fix src tests

# Format code
ruff format src tests

# Check formatting without changing files
ruff format --check src tests
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Type check the entire codebase
mypy src

# Type check with detailed output
mypy --show-error-codes src

# Type check specific files
mypy src/vllm_orchestrator/core/responses/
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before committing:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Code Quality Standards

- **Type hints**: All public functions must have type hints
- **Docstrings**: Use Google-style docstrings for public APIs
- **Error handling**: Use specific exception types, include context
- **Logging**: Use structured logging with appropriate levels
- **Testing**: Maintain >90% test coverage

#### Example Code Style

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ResponsesService:
    """Service for managing response generation and storage.

    Handles the complete lifecycle of responses including:
    - Context resolution from previous responses
    - Tool orchestration
    - Storage persistence

    Args:
        vllm_client: Client for communicating with vLLM
        responses_store: Storage backend for responses
        tool_orchestrator: Tool execution coordinator
    """

    def __init__(
        self,
        vllm_client: VllmHttpClient,
        responses_store: ResponsesStore,
        tool_orchestrator: Optional[ToolOrchestrator] = None
    ) -> None:
        self.vllm_client = vllm_client
        self.responses_store = responses_store
        self.tool_orchestrator = tool_orchestrator

    async def create_response(
        self,
        model: str,
        input: List[ResponseInputItemParam],
        *,
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_output_tokens: int = 1000
    ) -> Response:
        """Create a new response with optional tool execution.

        Args:
            model: Model identifier for generation
            input: Input messages or prompt
            previous_response_id: ID of previous response for chaining
            tools: Available tools for execution
            max_output_tokens: Maximum tokens to generate

        Returns:
            Generated response with metadata

        Raises:
            ResponseNotFoundError: If previous_response_id is invalid
            VllmClientError: If vLLM communication fails
            ToolExecutionError: If tool execution fails
        """
        try:
            # Implementation...
            pass
        except Exception as e:
            logger.error(
                "Failed to create response",
                model=model,
                previous_response_id=previous_response_id,
                error=str(e),
                exc_info=True
            )
            raise
```

## Contributing Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/vllm-orchestrator.git
cd vllm-orchestrator

# Add upstream remote
git remote add upstream https://github.com/your-org/vllm-orchestrator.git
```

### 2. Create a Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 3. Make Changes

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass

### 4. Commit Changes

We use conventional commit messages:

```bash
# Format: <type>(<scope>): <description>
git commit -m "feat(responses): add streaming support for tool calls"
git commit -m "fix(vectors): handle empty query results gracefully"
git commit -m "docs(api): add examples for file search endpoint"
git commit -m "test(tools): add integration tests for file search"
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/dependency updates

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Use the PR template to provide context
```

### 6. Code Review Process

- All PRs require at least one review
- Address feedback promptly
- Keep PRs focused and reasonably sized (<500 lines)
- Rebase if requested to maintain clean history

## Architecture Guidelines

### Core Principles

1. **Separation of Concerns**: API layer → Service layer → Storage layer
2. **Dependency Injection**: Use constructor injection for testability
3. **Async First**: All I/O operations should be async
4. **Type Safety**: Leverage Python's type system extensively
5. **Error Transparency**: Errors should bubble up with context

### Adding New API Endpoints

1. **Create the router** in `src/vllm_orchestrator/api/`
2. **Define Pydantic models** for request/response in `protocol.py`
3. **Implement service logic** in `src/vllm_orchestrator/core/`
4. **Add database models** in `src/vllm_orchestrator/storage/models.py`
5. **Write comprehensive tests**

Example structure:

```python
# api/new_feature.py
from fastapi import APIRouter, Depends
from vllm_orchestrator.core.new_feature.service import NewFeatureService

router = APIRouter(prefix="/v1/new-feature")

@router.post("/")
async def create_item(
    request: CreateItemRequest,
    service: NewFeatureService = Depends()
) -> CreateItemResponse:
    return await service.create_item(request)

# core/new_feature/service.py
class NewFeatureService:
    def __init__(self, store: NewFeatureStore):
        self.store = store

    async def create_item(self, request: CreateItemRequest) -> CreateItemResponse:
        # Business logic here
        pass
```

### Error Handling Patterns

```python
# Define specific exception types
class ResponseNotFoundError(Exception):
    """Raised when a response ID cannot be found."""

    def __init__(self, response_id: str):
        self.response_id = response_id
        super().__init__(f"Response not found: {response_id}")

# Service layer error handling
async def get_response(self, response_id: str) -> Response:
    try:
        response = await self.store.get_response(response_id)
        if response is None:
            raise ResponseNotFoundError(response_id)
        return response
    except DatabaseError as e:
        logger.error("Database error retrieving response", response_id=response_id, error=str(e))
        raise

# API layer error handling
@router.get("/{response_id}")
async def get_response(response_id: str, service: ResponsesService = Depends()):
    try:
        return await service.get_response(response_id)
    except ResponseNotFoundError:
        raise HTTPException(status_code=404, detail="Response not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Database Changes

### Creating Migrations

We use Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "add new table for feature X"

# Review the generated migration in alembic/versions/

# Apply migration
alembic upgrade head

# Downgrade if needed
alembic downgrade -1
```

### Model Guidelines

- Use descriptive table and column names
- Include created_at/updated_at timestamps
- Add appropriate indexes for query performance
- Use foreign key constraints
- Consider data types carefully (especially for JSON fields)

```python
class NewModel(Base):
    __tablename__ = "new_models"

    id = Column(String(64), primary_key=True, default=lambda: f"new_{uuid.uuid4().hex[:8]}")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Add indexes for common queries
    __table_args__ = (
        Index("idx_new_models_created_at", "created_at"),
    )
```

## Adding New Tools

Tools follow a plugin architecture using the `ToolExecutor` base class:

### 1. Create Tool Executor

```python
# tools/my_new_tool.py
from typing import Dict, Any
from vllm_orchestrator.tools.base import ToolExecutor, ToolResult

class MyNewToolExecutor(ToolExecutor):
    """Executor for my_new_tool functionality."""

    @property
    def name(self) -> str:
        return "my_new_tool"

    @property
    def description(self) -> str:
        return "Does something useful"

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        # Validate arguments
        if "required_param" not in arguments:
            return ToolResult(
                success=False,
                content="Missing required parameter: required_param"
            )

        try:
            # Tool implementation
            result = await self._do_work(arguments["required_param"])
            return ToolResult(
                success=True,
                content=f"Work completed: {result}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Tool execution failed: {str(e)}"
            )

    async def _do_work(self, param: str) -> str:
        # Actual tool logic
        return f"processed {param}"
```

### 2. Register Tool

```python
# In your application setup
tool_registry = ToolRegistry()
my_tool = MyNewToolExecutor()
tool_registry.register(my_tool)
```

### 3. Add Tests

```python
# tests/test_my_new_tool.py
import pytest
from vllm_orchestrator.tools.my_new_tool import MyNewToolExecutor

@pytest.fixture
def tool():
    return MyNewToolExecutor()

@pytest.mark.asyncio
async def test_tool_success(tool):
    result = await tool.execute({"required_param": "test_value"})
    assert result.success
    assert "processed test_value" in result.content

@pytest.mark.asyncio
async def test_tool_missing_param(tool):
    result = await tool.execute({})
    assert not result.success
    assert "Missing required parameter" in result.content
```

## Debugging

### Logging Configuration

Enable debug logging during development:

```bash
export LOG_LEVEL=DEBUG
export JSON_LOGS=false  # Human-readable logs for development

python -m vllm_orchestrator.main
```

### Database Debugging

```bash
# Enable SQL query logging
export SQLALCHEMY_ECHO=true

# Or in code
engine = create_engine(database_url, echo=True)
```

### vLLM Client Debugging

```bash
# Enable HTTP request logging
export VLLM_CLIENT_DEBUG=true

# Test vLLM connectivity
curl http://localhost:8001/v1/models
```

### Common Debugging Scenarios

**1. Test Failures**

```bash
# Run single failing test with verbose output
pytest tests/test_responses.py::test_create_response -v -s

# Debug with pdb
pytest tests/test_responses.py::test_create_response --pdb

# Show test coverage for specific file
pytest --cov=vllm_orchestrator.core.responses tests/test_responses.py
```

**2. Database Issues**

```bash
# Check database connection
python -c "
from vllm_orchestrator.storage.database import create_engine
engine = create_engine()
with engine.connect() as conn:
    result = conn.execute('SELECT 1')
    print('Database OK')
"

# Inspect tables
python -c "
from vllm_orchestrator.storage.database import create_engine
from sqlalchemy import inspect
engine = create_engine()
inspector = inspect(engine)
print(inspector.get_table_names())
"
```

**3. Integration Debugging**

```bash
# Start services with debug logging
docker-compose up postgres redis
LOG_LEVEL=DEBUG python -m vllm_orchestrator.main

# In another terminal, test API
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "input": "Hello world"}' \
  -v
```

## Performance Testing

### Load Testing

Run comprehensive load tests to verify performance:

```bash
# Start the orchestrator
python -m vllm_orchestrator.main

# In another terminal, run load tests
python tests/load_test.py \
  --url http://localhost:8000 \
  --duration 300 \
  --rps 5 \
  --concurrent 10

# Monitor resource usage
docker stats  # If using Docker
htop          # System monitoring
```

### Benchmarking

```bash
# Run specific performance tests
pytest tests/integration/test_performance.py -v

# Profile memory usage
python -m memory_profiler tests/profile_example.py

# Profile CPU usage
python -m cProfile -o profile.stats tests/profile_example.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('time').print_stats(10)"
```

## Troubleshooting Guide

### Common Development Issues

**1. Import Errors**
```bash
# Issue: Module not found errors
# Solution: Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import vllm_orchestrator; print('✅ Import OK')"
```

**2. Database Connection Issues**
```bash
# Issue: Connection refused to PostgreSQL
# Check if PostgreSQL is running
docker-compose ps postgres
# Restart if needed
docker-compose restart postgres

# Issue: Missing tables
# Run migrations
python -m vllm_orchestrator.storage.migrations

# Reset database completely
docker-compose down -v  # WARNING: Deletes all data
docker-compose up -d postgres
python -m vllm_orchestrator.storage.migrations
```

**3. vLLM Connection Issues**
```bash
# Issue: Cannot connect to vLLM server
# Check if vLLM is running
curl http://localhost:8001/health

# Common vLLM startup command
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-medium \
  --port 8001 \
  --host 0.0.0.0

# For GPU users
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --port 8001 \
  --tensor-parallel-size 2  # Adjust based on GPU count
```

**4. Redis Issues**
```bash
# Issue: Redis connection failed
# Check Redis status
docker-compose ps redis
redis-cli ping  # Should return "PONG"

# Restart Redis
docker-compose restart redis

# Clear Redis cache
redis-cli FLUSHALL
```

**5. Test Failures**
```bash
# Issue: Tests failing due to environment
# Clean test environment
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up -d

# Run tests in isolation
pytest tests/unit/  # No external dependencies
pytest tests/integration/ --tb=short  # With external services

# Debug specific test
pytest tests/test_responses.py::test_create_response -vvs --pdb
```

## Development Tools and Tips

### Useful Make Commands

Create a `Makefile` for common tasks:

```makefile
.PHONY: install dev-install test lint format typecheck clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=vllm_orchestrator --cov-report=html

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
	rm -rf .coverage htmlcov/

all: clean format lint typecheck test
```

### Environment Management

Use `.env` files for configuration:

```bash
# .env.development
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vllm_orchestrator_dev
REDIS_URL=redis://localhost:6379
VLLM_BASE_URL=http://localhost:8001
LOG_LEVEL=DEBUG
JSON_LOGS=false

# .env.test
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vllm_orchestrator_test
REDIS_URL=redis://localhost:6379/1
VLLM_BASE_URL=http://localhost:8002
LOG_LEVEL=INFO

# Load environment
export $(grep -v '^#' .env.development | xargs)
```

## Contributing Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines (ruff format + ruff check)
- [ ] Type hints are present and mypy passes
- [ ] All tests pass (pytest)
- [ ] Test coverage is maintained (>90%)
- [ ] Documentation is updated for user-facing changes
- [ ] Commit messages follow conventional format
- [ ] PR description explains what and why
- [ ] No sensitive information (API keys, passwords) in code
- [ ] Database migrations are included if schema changes
- [ ] Performance impact is considered for large changes

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Check the Wiki for detailed guides

## License

By contributing to vLLM Orchestrator, you agree that your contributions will be licensed under the Apache 2.0 License.