# vLLM Orchestrator

A production-ready stateful orchestration layer for vLLM that provides OpenAI-compatible APIs with file search (RAG), tool execution, and persistent storage.

![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/vllm-orchestrator/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│        OpenAI SDK / Custom Clients / Agents / Applications       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ OpenAI-Compatible REST API
┌─────────────────────────────────────────────────────────────────┐
│              STATEFUL GATEWAY ("vLLM Orchestrator")              │
│                                                                  │
│  • Responses Router      • Files & Vector Stores                 │
│  • Tool Orchestration    • Persistent Storage (PostgreSQL)       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ Internal API (HTTP)
┌─────────────────────────────────────────────────────────────────┐
│                      VLLM INFERENCE CLUSTER                      │
│                                                                  │
│  • Token-based scheduling    • Prefix caching                    │
│  • store=False (stateless)   • Pure inference                    │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- ✨ **Stateful Responses API**: OpenAI-compatible responses with conversation chaining
- 🔍 **File Search (RAG)**: Upload documents and search them semantically using vector stores
- 🛠️ **Tool Orchestration**: Multi-turn conversations with tool calls (file_search, web_search)
- 🏭 **Production Ready**: Observability, rate limiting, security, autoscaling
- 💾 **Storage Options**: SQLite (dev) or PostgreSQL with pgvector (production)
- ⚡ **Caching**: Redis-based caching and rate limiting
- 🔌 **OpenAI SDK Compatible**: Drop-in replacement for OpenAI API
- 🚀 **High Performance**: Optimized connection pooling and async processing
- 📊 **Monitoring**: Prometheus metrics and structured logging
- 🔒 **Security**: API key authentication, rate limiting, CORS protection

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-orchestrator.git
cd vllm-orchestrator

# Start all services (PostgreSQL, Redis, vLLM, Orchestrator)
docker-compose up -d

# Check logs
docker-compose logs -f orchestrator

# The API will be available at http://localhost:8000
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start required services
docker-compose up -d postgres redis

# Start vLLM server (in another terminal)
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --port 8001

# Set environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vllm_orchestrator"
export REDIS_URL="redis://localhost:6379"
export VLLM_BASE_URL="http://localhost:8001"

# Run database migrations
python -m vllm_orchestrator.storage.migrations

# Start the orchestrator
python -m vllm_orchestrator.main
```

### Option 3: Quick SQLite Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Start with SQLite (no external dependencies)
export DATABASE_URL="sqlite+aiosqlite:///./orchestrator.db"
export VLLM_BASE_URL="http://localhost:8001"  # Point to your vLLM instance

# Run the server
python -m vllm_orchestrator.main
```

## Quick Live Demo

```bash
# Terminal 1: Start vLLM backend
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-medium \
  --port 8001

# Terminal 2: Start orchestrator
python -m vllm_orchestrator.main

# Terminal 3: Run live demo
python demo_live_test.py
```

Tests conversation memory, streaming, and RAG with real models.

## Sample Applications

### 1. Basic Response Generation

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Simple response
response = client.post("/v1/responses", json={
    "model": "microsoft/DialoGPT-medium",
    "input": "Hello! How are you today?",
    "max_output_tokens": 100
})

print(response.json()["output"]["content"])
```

### 2. Multi-Turn Conversations

```python
# First message
response1 = client.post("/v1/responses", json={
    "model": "microsoft/DialoGPT-medium",
    "input": "My name is Alice and I love machine learning.",
    "max_output_tokens": 100
})

response_id = response1.json()["id"]

# Follow-up with context
response2 = client.post("/v1/responses", json={
    "model": "microsoft/DialoGPT-medium",
    "input": "What did I just tell you about myself?",
    "previous_response_id": response_id,  # This chains the conversation
    "max_output_tokens": 100
})

print(response2.json()["output"]["content"])
# Output: "You mentioned that your name is Alice and you love machine learning!"
```

### 3. File Search (RAG)

```python
from openai import OpenAI

# Use OpenAI client for compatibility
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Upload document
with open("research_paper.pdf", "rb") as f:
    file = client.files.create(file=f, purpose="assistants")

# Create vector store
vector_store = client.vector_stores.create(name="research_papers")
client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

# Search with file_search tool
response = client.responses.create(
    model="microsoft/DialoGPT-medium",
    input="What are the key findings about transformer architectures?",
    tools=[{"type": "file_search", "file_search": {"vector_store_ids": [vector_store.id]}}],
    max_output_tokens=300
)

print(response.output[0].content[0].text)
# The model searches the document and provides contextual answers
```

### 4. Streaming Responses

```python
import httpx
import asyncio

async def stream_response():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/responses",
            json={
                "model": "microsoft/DialoGPT-medium",
                "input": "Tell me a story about a robot",
                "stream": True,
                "max_output_tokens": 200
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        import json
                        event = json.loads(data)
                        if event["type"] == "content":
                            print(event["delta"], end="", flush=True)

# Run with: asyncio.run(stream_response())
```

### 5. OpenAI SDK Compatibility

```python
from openai import OpenAI

# Point OpenAI SDK to vLLM Orchestrator
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required for local dev
)

# Use standard OpenAI SDK methods
completion = client.chat.completions.create(
    model="microsoft/DialoGPT-medium",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100
)

print(completion.choices[0].message.content)
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/vllm_orchestrator
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis (optional - disables caching if not set)
REDIS_URL=redis://localhost:6379
REDIS_DEFAULT_TTL=3600

# vLLM Connection
VLLM_BASE_URL=http://localhost:8001
VLLM_API_KEY=optional_api_key

# Security (production)
REQUIRE_API_KEY=false
API_KEYS=sk-key1,sk-key2
MAX_REQUEST_SIZE=10485760  # 10MB

# Rate Limiting (requires Redis)
ENABLE_RATE_LIMITING=false
GLOBAL_REQUESTS_PER_MINUTE=1000
TENANT_REQUESTS_PER_MINUTE=100

# Monitoring
ENABLE_METRICS=true
ENABLE_DB_MONITORING=true
LOG_LEVEL=INFO
JSON_LOGS=false

# Performance
WORKERS=4
```

### Configuration File

Create `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

database:
  url: "postgresql://user:pass@localhost:5432/vllm_orchestrator"
  pool_size: 20
  max_overflow: 30

vllm:
  base_url: "http://localhost:8001"
  timeout: 300
  max_retries: 3

security:
  require_api_key: false
  api_keys: ["sk-key1", "sk-key2"]
  cors_origins: ["http://localhost:3000"]

rate_limiting:
  enable: false
  global_rpm: 1000
  tenant_rpm: 100

monitoring:
  enable_metrics: true
  enable_db_monitoring: true
```

## API Reference

### Responses API

#### Create Response
```bash
POST /v1/responses
```

**Request:**
```json
{
  "model": "microsoft/DialoGPT-medium",
  "input": "Hello world",
  "max_output_tokens": 100,
  "temperature": 0.7,
  "tools": [{"type": "file_search"}],
  "previous_response_id": "resp_abc123",
  "stream": false
}
```

**Response:**
```json
{
  "id": "resp_def456",
  "created_at": "2024-01-15T10:30:00Z",
  "model": "microsoft/DialoGPT-medium",
  "status": "completed",
  "input": {...},
  "output": {
    "content": "Hello! How can I help you today?",
    "token_count": 8
  },
  "usage": {
    "input_tokens": 2,
    "output_tokens": 8,
    "total_tokens": 10
  }
}
```

#### Get Response
```bash
GET /v1/responses/{response_id}
```

### Files API

#### Upload File
```bash
POST /v1/files
Content-Type: multipart/form-data

file: <file_content>
purpose: assistants
```

#### List Files
```bash
GET /v1/files?purpose=assistants&limit=10
```

### Vector Stores API

#### Create Vector Store
```bash
POST /v1/vector_stores
```

```json
{
  "name": "research_papers",
  "metadata": {"description": "Collection of research papers"}
}
```

#### Search Vector Store
```bash
POST /v1/vector_stores/{vector_store_id}/search
```

```json
{
  "query": "transformer architecture",
  "max_results": 5,
  "score_threshold": 0.7
}
```

## Production Deployment

### Docker Production Stack

```bash
# Deploy with Docker Compose
cd deployment
docker-compose -f docker-compose.production.yml up -d

# Scale services
docker-compose -f docker-compose.production.yml up -d --scale orchestrator=3

# View logs
docker-compose -f docker-compose.production.yml logs -f orchestrator

# Monitor with Prometheus/Grafana
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana (admin/admin)
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -l app=vllm-orchestrator
kubectl get hpa vllm-orchestrator-hpa

# View logs
kubectl logs -l app=vllm-orchestrator --tail=100 -f

# Scale manually
kubectl scale deployment vllm-orchestrator --replicas=5
```

### Production Checklist

- [ ] Set `REQUIRE_API_KEY=true` and configure `API_KEYS`
- [ ] Use PostgreSQL with pgvector extension
- [ ] Configure Redis for caching and rate limiting
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (ELK stack)
- [ ] Set resource limits in Kubernetes
- [ ] Configure backups for PostgreSQL
- [ ] Set up SSL/TLS termination at load balancer
- [ ] Configure network policies and security groups

## Monitoring

### Health Checks

- `GET /health` - Detailed component health
- `GET /ready` - Kubernetes readiness probe
- `GET /live` - Kubernetes liveness probe
- `GET /metrics` - Prometheus metrics

### Key Metrics

- Response generation latency (P95 < 2s target)
- Tool execution success rate (> 99% target)
- Database connection pool utilization (< 80%)
- Redis cache hit rate (> 90% target)
- Request throughput (requests/second)

### Alerts

- **Critical**: Response time P95 > 5 seconds
- **Warning**: Error rate > 5% for 5 minutes
- **Info**: New deployment completed

## Performance

### Load Testing

```bash
# Install test dependencies
pip install httpx asyncio

# Run load test
python tests/load_test.py \
  --url http://localhost:8000 \
  --duration 300 \
  --rps 10 \
  --concurrent 20

# Monitor during load test
kubectl top pods -l app=vllm-orchestrator
```

## Troubleshooting

### Common Issues

**1. vLLM Connection Failed**
```bash
# Check vLLM server status
curl http://localhost:8001/health

# Check environment variable
echo $VLLM_BASE_URL
```

**2. Database Connection Issues**
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Test database connection
python -c "
from vllm_orchestrator.storage.database import create_engine
engine = create_engine()
print('Database connected successfully')
"
```

**3. Vector Search Not Working**
```bash
# Check pgvector extension
docker-compose exec postgres psql -U postgres -d vllm_orchestrator -c "SELECT * FROM pg_extension WHERE extname='vector';"

# Recreate vector store
curl -X DELETE http://localhost:8000/v1/vector_stores/{id}
curl -X POST http://localhost:8000/v1/vector_stores -H "Content-Type: application/json" -d '{"name": "test"}'
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export JSON_LOGS=true

# Run with debug
python -m vllm_orchestrator.main

# Check detailed logs
tail -f logs/vllm-orchestrator.log | jq .
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and contribution workflow.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=vllm_orchestrator

# Lint and format
ruff check src tests
ruff format src tests

# Type checking
mypy src
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Running tests and quality checks
- Submitting pull requests
- Coding standards and best practices

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/vllm-orchestrator/issues)
- **Documentation**: [Wiki](https://github.com/your-org/vllm-orchestrator/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/vllm-orchestrator/discussions)
