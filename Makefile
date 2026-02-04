.PHONY: install dev-install test lint format typecheck clean all help start-services stop-services reset-db

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install package in normal mode"
	@echo "  dev-install   Install package with dev dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code"
	@echo "  typecheck     Run type checking"
	@echo "  clean         Clean build artifacts and cache"
	@echo "  all           Run full quality pipeline"
	@echo "  start-services Start PostgreSQL and Redis with Docker"
	@echo "  stop-services  Stop Docker services"
	@echo "  reset-db      Reset database (WARNING: deletes all data)"

# Installation
install:
	uv sync

dev-install:
	uv sync --extra dev

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=vllm_orchestrator --cov-report=html --cov-report=term

test-unit:
	uv run pytest tests/unit/

test-integration:
	uv run pytest tests/integration/

# Code quality
lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

format-check:
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

# Development services
start-services:
	docker-compose up -d postgres redis

stop-services:
	docker-compose down

reset-db:
	@echo "WARNING: This will delete all data in the database!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker-compose down -v; \
		docker-compose up -d postgres redis; \
		sleep 5; \
		uv run python -m vllm_orchestrator.storage.migrations; \
		echo "Database reset complete"; \
	else \
		echo "Database reset cancelled"; \
	fi

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
	rm -rf .coverage htmlcov/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/

# Full quality pipeline
all: clean format lint typecheck test

# Development workflow
dev: dev-install start-services
	@echo "Development environment ready!"
	@echo "Run 'make reset-db' to initialize the database"
	@echo "Then start the orchestrator with: uv run python -m vllm_orchestrator.main"

# Production build
build:
	docker build -t vllm-orchestrator -f deployment/Dockerfile .

# Load testing
load-test:
	@echo "Starting load test (make sure orchestrator is running on http://localhost:8000)"
	uv run python tests/load_test.py --url http://localhost:8000 --duration 60 --rps 5 --concurrent 5