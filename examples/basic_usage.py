#!/usr/bin/env python3
"""Basic usage examples for vLLM Orchestrator.

This script demonstrates the core functionality:
- Simple response generation
- Multi-turn conversations with previous_response_id
- Basic error handling
"""

import asyncio
import httpx
from typing import Optional


class VllmOrchestratorClient:
    """Simple client for vLLM Orchestrator API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["x-api-key"] = api_key

    async def create_response(
        self,
        model: str,
        input: str,
        *,
        previous_response_id: Optional[str] = None,
        max_output_tokens: int = 100,
        temperature: float = 0.7
    ) -> dict:
        """Create a new response."""
        payload = {
            "model": model,
            "input": input,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        response = await self.client.post(
            f"{self.base_url}/v1/responses",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    async def get_response(self, response_id: str) -> dict:
        """Get an existing response."""
        response = await self.client.get(
            f"{self.base_url}/v1/responses/{response_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def example_1_simple_response():
    """Example 1: Simple response generation."""
    print("🤖 Example 1: Simple Response Generation")
    print("-" * 50)

    client = VllmOrchestratorClient()

    try:
        response = await client.create_response(
            model="microsoft/DialoGPT-medium",
            input="Hello! What's the weather like today?",
            max_output_tokens=50
        )

        print(f"Response ID: {response['id']}")
        print(f"Status: {response['status']}")
        print(f"Content: {response['output']['content']}")
        print(f"Tokens used: {response['usage']['total_tokens']}")

    except httpx.HTTPStatusError as e:
        print(f"❌ API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await client.close()

    print()


async def example_2_multi_turn_conversation():
    """Example 2: Multi-turn conversation using previous_response_id."""
    print("💬 Example 2: Multi-Turn Conversation")
    print("-" * 50)

    client = VllmOrchestratorClient()

    try:
        # First message
        response1 = await client.create_response(
            model="microsoft/DialoGPT-medium",
            input="Hi! My name is Alice and I'm a software engineer.",
            max_output_tokens=50
        )

        print("👤 User: Hi! My name is Alice and I'm a software engineer.")
        print(f"🤖 Assistant: {response1['output']['content']}")
        print()

        # Second message - referencing previous conversation
        response2 = await client.create_response(
            model="microsoft/DialoGPT-medium",
            input="What did I just tell you about myself?",
            previous_response_id=response1["id"],  # This maintains conversation context
            max_output_tokens=50
        )

        print("👤 User: What did I just tell you about myself?")
        print(f"🤖 Assistant: {response2['output']['content']}")
        print()

        # Third message - continuing the conversation
        response3 = await client.create_response(
            model="microsoft/DialoGPT-medium",
            input="What programming languages do you recommend for beginners?",
            previous_response_id=response2["id"],
            max_output_tokens=100
        )

        print("👤 User: What programming languages do you recommend for beginners?")
        print(f"🤖 Assistant: {response3['output']['content']}")

        print(f"\n📊 Conversation Stats:")
        print(f"   Total responses: 3")
        print(f"   Total tokens: {sum(r['usage']['total_tokens'] for r in [response1, response2, response3])}")

    except httpx.HTTPStatusError as e:
        print(f"❌ API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await client.close()

    print()


async def example_3_response_retrieval():
    """Example 3: Retrieving stored responses."""
    print("📚 Example 3: Response Retrieval")
    print("-" * 50)

    client = VllmOrchestratorClient()

    try:
        # Create a response
        response = await client.create_response(
            model="microsoft/DialoGPT-medium",
            input="Tell me a fun fact about Python programming.",
            max_output_tokens=80
        )

        response_id = response["id"]
        print(f"✅ Created response: {response_id}")
        print(f"Content: {response['output']['content'][:100]}...")
        print()

        # Retrieve the same response
        retrieved = await client.get_response(response_id)

        print(f"📖 Retrieved response: {retrieved['id']}")
        print(f"Created at: {retrieved['created_at']}")
        print(f"Status: {retrieved['status']}")
        print(f"Model: {retrieved['model']}")
        print(f"Full content: {retrieved['output']['content']}")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print("❌ Response not found")
        else:
            print(f"❌ API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await client.close()

    print()


async def example_4_error_handling():
    """Example 4: Error handling."""
    print("⚠️ Example 4: Error Handling")
    print("-" * 50)

    client = VllmOrchestratorClient()

    try:
        # Try to get a non-existent response
        print("Attempting to retrieve non-existent response...")
        try:
            await client.get_response("resp_nonexistent")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print("✅ Correctly handled 404 - Response not found")
            else:
                print(f"❌ Unexpected error: {e.response.status_code}")

        print()

        # Try to use invalid previous_response_id
        print("Attempting to use invalid previous_response_id...")
        try:
            await client.create_response(
                model="microsoft/DialoGPT-medium",
                input="Hello",
                previous_response_id="resp_invalid"
            )
        except httpx.HTTPStatusError as e:
            print(f"✅ Correctly handled error: {e.response.status_code}")
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Error detail: {error_detail}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        await client.close()

    print()


async def example_5_health_check():
    """Example 5: Health check."""
    print("🏥 Example 5: Health Check")
    print("-" * 50)

    client = VllmOrchestratorClient()

    try:
        # Check service health
        response = await client.client.get(f"{client.base_url}/health")
        health_data = response.json()

        print("🔍 Service Health Status:")
        print(f"   Overall status: {health_data.get('status', 'unknown')}")

        if "components" in health_data:
            for component, status in health_data["components"].items():
                status_icon = "✅" if status.get("status") == "healthy" else "❌"
                print(f"   {status_icon} {component}: {status.get('status', 'unknown')}")

        if "database" in health_data:
            db_info = health_data["database"]
            print(f"\n📊 Database Info:")
            print(f"   Connection pool: {db_info.get('pool_size', 'unknown')} / {db_info.get('pool_max', 'unknown')}")

    except Exception as e:
        print(f"❌ Health check failed: {e}")

    finally:
        await client.close()

    print()


async def main():
    """Run all examples."""
    print("🚀 vLLM Orchestrator - Basic Usage Examples")
    print("=" * 60)
    print()

    # Check if the service is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code != 200:
                print("❌ vLLM Orchestrator is not responding properly")
                print("   Make sure the service is running on http://localhost:8000")
                return
    except Exception:
        print("❌ Cannot connect to vLLM Orchestrator")
        print("   Make sure the service is running on http://localhost:8000")
        print("   Start with: python -m vllm_orchestrator.main")
        return

    print("✅ vLLM Orchestrator is running!")
    print()

    # Run examples
    await example_1_simple_response()
    await example_2_multi_turn_conversation()
    await example_3_response_retrieval()
    await example_4_error_handling()
    await example_5_health_check()

    print("🎉 All examples completed!")
    print("\n📚 Next steps:")
    print("   - Try the file search example: python examples/file_search_example.py")
    print("   - Try streaming: python examples/streaming_example.py")
    print("   - Read the documentation: https://github.com/your-org/vllm-orchestrator")


if __name__ == "__main__":
    asyncio.run(main())