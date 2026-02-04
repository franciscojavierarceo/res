#!/usr/bin/env python3
"""Development server runner for vLLM Orchestrator.

This script provides an easy way to start the development server with
appropriate configuration for local development and testing.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import vllm_orchestrator
        print("✅ vLLM Orchestrator package found")
        return True
    except ImportError:
        print("❌ vLLM Orchestrator not installed")
        print("   Run: pip install -e .[dev]")
        return False


def check_vllm_server():
    """Check if vLLM server is running."""
    import httpx

    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001")

    try:
        with httpx.Client() as client:
            response = client.get(f"{vllm_url}/v1/models", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_names = [model.get("id", "unknown") for model in models]
                print(f"✅ vLLM server running at {vllm_url}")
                print(f"   Available models: {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}")
                return True
            else:
                print(f"❌ vLLM server responded with status {response.status_code}")
                return False

    except Exception as e:
        print(f"❌ Cannot connect to vLLM server at {vllm_url}")
        print(f"   Error: {e}")
        print("\n💡 To start a vLLM server:")
        print("   pip install vllm")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("       --model microsoft/DialoGPT-medium \\")
        print("       --port 8001")
        return False


def setup_environment():
    """Set up environment variables for development."""
    # Database URL
    if not os.getenv("DATABASE_URL"):
        db_url = "sqlite+aiosqlite:///./dev_orchestrator.db"
        os.environ["DATABASE_URL"] = db_url
        print(f"📁 Using SQLite database: {db_url}")

    # vLLM URL
    if not os.getenv("VLLM_BASE_URL"):
        vllm_url = "http://localhost:8001"
        os.environ["VLLM_BASE_URL"] = vllm_url
        print(f"🔗 vLLM server URL: {vllm_url}")

    # Redis (optional)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print(f"⚡ Redis enabled: {redis_url}")
    else:
        print("⚠️  Redis not configured (caching disabled)")

    # Development settings
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("JSON_LOGS", "false")
    os.environ.setdefault("ENABLE_METRICS", "true")

    print(f"📊 Log level: {os.environ['LOG_LEVEL']}")


def check_database():
    """Check database connectivity and run migrations if needed."""
    try:
        # Import here to ensure environment is set up first
        from vllm_orchestrator.storage.database import create_engine

        # Test database connection
        engine = create_engine()
        print("✅ Database connection successful")

        # Check if we need to run migrations
        # This is a simple check - in production you'd use proper migration tools
        try:
            import sqlite3
            from pathlib import Path

            db_url = os.environ.get("DATABASE_URL", "")
            if "sqlite" in db_url:
                # Extract the database file path
                db_file = db_url.split("///")[-1] if ":///" in db_url else "dev_orchestrator.db"
                if not Path(db_file).exists():
                    print("🔧 Setting up new SQLite database...")
                else:
                    print(f"📁 Using existing database: {db_file}")

        except ImportError:
            pass  # Not SQLite

        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Database troubleshooting:")
        print("   - For SQLite: Check file permissions")
        print("   - For PostgreSQL: Ensure server is running and credentials are correct")
        print("   - Check DATABASE_URL environment variable")
        return False


def main():
    """Main entry point."""
    print("🚀 vLLM Orchestrator Development Server")
    print("=" * 50)
    print()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Set up environment
    setup_environment()
    print()

    # Check external dependencies
    dependencies_ok = True

    # Check vLLM server
    if not check_vllm_server():
        dependencies_ok = False

    print()

    # Check database
    if not check_database():
        dependencies_ok = False

    print()

    if not dependencies_ok:
        print("❌ Some dependencies are not available")
        print("   Please fix the issues above and try again")
        return 1

    # Start the server
    print("🎯 Starting vLLM Orchestrator...")
    print("   Server will be available at: http://localhost:8000")
    print("   Health check: http://localhost:8000/health")
    print("   API docs: http://localhost:8000/docs")
    print()
    print("📝 Logs will appear below:")
    print("-" * 50)

    try:
        # Import and run the main application
        from vllm_orchestrator.main import create_app
        import uvicorn

        app = create_app()

        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            reload=True,  # Auto-reload on code changes
            reload_dirs=["src"],  # Watch for changes in source directory
        )

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down vLLM Orchestrator...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())