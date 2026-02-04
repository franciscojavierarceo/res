#!/usr/bin/env python3
"""Streaming responses example for vLLM Orchestrator.

This script demonstrates:
- Real-time streaming of response generation
- Server-Sent Events (SSE) handling
- Streaming with tool calls
- Different streaming event types
"""

import asyncio
import json
import httpx
from typing import Optional, AsyncIterator


class StreamingVllmClient:
    """Client for streaming responses from vLLM Orchestrator."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["x-api-key"] = api_key

    async def stream_response(
        self,
        model: str,
        input: str,
        *,
        previous_response_id: Optional[str] = None,
        tools: Optional[list] = None,
        max_output_tokens: int = 200,
        temperature: float = 0.7
    ) -> AsyncIterator[dict]:
        """Stream a response with real-time updates."""
        payload = {
            "model": model,
            "input": input,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": True  # Enable streaming
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/responses",
                json=payload,
                headers=self.headers
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)
                            yield event
                        except json.JSONDecodeError:
                            continue


async def example_1_basic_streaming():
    """Example 1: Basic streaming response."""
    print("🔄 Example 1: Basic Streaming Response")
    print("-" * 50)

    client = StreamingVllmClient()

    print("👤 User: Tell me a short story about a robot learning to paint")
    print("🤖 Assistant: ", end="", flush=True)

    try:
        full_content = ""
        event_count = 0

        async for event in client.stream_response(
            model="microsoft/DialoGPT-medium",
            input="Tell me a short story about a robot learning to paint",
            max_output_tokens=150
        ):
            event_count += 1

            if event["type"] == "content":
                content_delta = event.get("delta", "")
                print(content_delta, end="", flush=True)
                full_content += content_delta

            elif event["type"] == "response.created":
                response_id = event.get("response_id")
                print(f"\n\n📋 Response ID: {response_id}")

            elif event["type"] == "response.completed":
                usage = event.get("usage", {})
                print(f"\n\n📊 Streaming completed:")
                print(f"   Events received: {event_count}")
                print(f"   Total tokens: {usage.get('total_tokens', 'unknown')}")
                print(f"   Content length: {len(full_content)} characters")

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")

    print("\n")


async def example_2_streaming_with_tools():
    """Example 2: Streaming with tool calls."""
    print("🛠️ Example 2: Streaming with Tool Calls")
    print("-" * 50)

    # First, let's create a simple vector store for this example
    print("📚 Setting up vector store for demonstration...")

    try:
        # Create a sample document
        import tempfile
        from pathlib import Path

        sample_content = """
        Artificial Intelligence and Machine Learning

        AI is rapidly transforming industries through:
        - Natural language processing
        - Computer vision
        - Predictive analytics
        - Automated decision making

        Recent breakthroughs include large language models like GPT and BERT,
        which can understand and generate human-like text.
        """

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_ai_doc.txt', delete=False)
        temp_file.write(sample_content)
        temp_file.flush()

        # Upload file and create vector store
        async with httpx.AsyncClient() as http_client:
            # Upload file
            with open(temp_file.name, 'rb') as f:
                files = {"file": (Path(temp_file.name).name, f, "text/plain")}
                data = {"purpose": "assistants"}
                upload_response = await http_client.post(
                    "http://localhost:8000/v1/files",
                    files=files,
                    data=data
                )
            file_data = upload_response.json()

            # Create vector store
            vs_response = await http_client.post(
                "http://localhost:8000/v1/vector_stores",
                json={"name": "ai_knowledge", "file_ids": [file_data["id"]]},
                headers={"Content-Type": "application/json"}
            )
            vector_store_data = vs_response.json()
            vector_store_id = vector_store_data["id"]

        # Clean up temp file
        Path(temp_file.name).unlink()

        print(f"✅ Vector store ready: {vector_store_id}")

    except Exception as e:
        print(f"❌ Could not set up vector store: {e}")
        print("   Continuing with basic streaming example...")
        return

    print()

    client = StreamingVllmClient()

    print("👤 User: What are the latest trends in AI? Use the documents to help answer.")
    print("🤖 Assistant: ", end="", flush=True)

    try:
        tool_calls_made = 0
        content_parts = []

        async for event in client.stream_response(
            model="microsoft/DialoGPT-medium",
            input="What are the latest trends in AI? Use the documents to help answer.",
            tools=[{
                "type": "file_search",
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }],
            max_output_tokens=200
        ):
            event_type = event.get("type", "unknown")

            if event_type == "content":
                content_delta = event.get("delta", "")
                print(content_delta, end="", flush=True)
                content_parts.append(content_delta)

            elif event_type == "tool_call":
                tool_calls_made += 1
                tool_name = event.get("tool_name", "unknown")
                print(f"\n\n🔧 Tool call #{tool_calls_made}: {tool_name}")
                print("   Searching documents...", end="", flush=True)

            elif event_type == "tool_result":
                tool_name = event.get("tool_name", "unknown")
                success = event.get("success", False)
                status_icon = "✅" if success else "❌"
                print(f" {status_icon}")

                if success:
                    results_count = len(event.get("results", []))
                    print(f"   Found {results_count} relevant document chunks")
                else:
                    print(f"   Tool execution failed: {event.get('error', 'Unknown error')}")

                print("\n🤖 Continuing response: ", end="", flush=True)

            elif event_type == "response.completed":
                usage = event.get("usage", {})
                print(f"\n\n📊 Streaming completed:")
                print(f"   Tool calls made: {tool_calls_made}")
                print(f"   Total tokens: {usage.get('total_tokens', 'unknown')}")
                print(f"   Response length: {len(''.join(content_parts))} characters")

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")

    print("\n")


async def example_3_streaming_conversation():
    """Example 3: Streaming multi-turn conversation."""
    print("💬 Example 3: Streaming Multi-Turn Conversation")
    print("-" * 50)

    client = StreamingVllmClient()

    conversation = [
        "Hi! I'm learning about programming. Can you help me?",
        "What's the difference between Python and JavaScript?",
        "Which one would be better for data analysis?"
    ]

    previous_response_id = None

    for i, message in enumerate(conversation, 1):
        print(f"👤 Turn {i}: {message}")
        print(f"🤖 Assistant: ", end="", flush=True)

        try:
            response_id = None

            async for event in client.stream_response(
                model="microsoft/DialoGPT-medium",
                input=message,
                previous_response_id=previous_response_id,
                max_output_tokens=100
            ):
                if event["type"] == "content":
                    print(event.get("delta", ""), end="", flush=True)

                elif event["type"] == "response.created":
                    response_id = event.get("response_id")

                elif event["type"] == "response.completed":
                    print()  # New line after response

            # Update for next turn
            previous_response_id = response_id

        except Exception as e:
            print(f"\n❌ Error in turn {i}: {e}")
            break

        print()  # Extra line between turns

    print("✅ Multi-turn streaming conversation completed!")
    print()


async def example_4_streaming_with_error_handling():
    """Example 4: Streaming with comprehensive error handling."""
    print("⚠️ Example 4: Streaming with Error Handling")
    print("-" * 50)

    client = StreamingVllmClient()

    test_cases = [
        {
            "name": "Normal streaming",
            "input": "Count to 5",
            "should_fail": False
        },
        {
            "name": "Invalid previous response ID",
            "input": "Hello",
            "previous_response_id": "resp_invalid",
            "should_fail": True
        },
        {
            "name": "Very long input",
            "input": "Tell me about " + "AI " * 1000,  # Might hit token limits
            "should_fail": True
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}: {test_case['name']}")

        try:
            kwargs = {
                "model": "microsoft/DialoGPT-medium",
                "input": test_case["input"],
                "max_output_tokens": 50
            }

            if "previous_response_id" in test_case:
                kwargs["previous_response_id"] = test_case["previous_response_id"]

            content_received = False

            async for event in client.stream_response(**kwargs):
                if event["type"] == "content":
                    content_received = True
                    if not test_case["should_fail"]:
                        print("   📝 Content received:", event.get("delta", "")[:50] + "...")

                elif event["type"] == "error":
                    print(f"   ❌ Error event: {event.get('error', 'Unknown error')}")

                elif event["type"] == "response.completed":
                    if test_case["should_fail"]:
                        print("   ⚠️ Unexpected success (test expected failure)")
                    else:
                        print("   ✅ Completed successfully")

            if test_case["should_fail"] and content_received:
                print("   ⚠️ Expected failure but received content")
            elif not test_case["should_fail"] and not content_received:
                print("   ⚠️ Expected content but none received")

        except httpx.HTTPStatusError as e:
            if test_case["should_fail"]:
                print(f"   ✅ Expected HTTP error: {e.response.status_code}")
            else:
                print(f"   ❌ Unexpected HTTP error: {e.response.status_code}")

        except Exception as e:
            if test_case["should_fail"]:
                print(f"   ✅ Expected error: {type(e).__name__}")
            else:
                print(f"   ❌ Unexpected error: {e}")

        print()


async def example_5_streaming_performance():
    """Example 5: Streaming performance measurement."""
    print("⚡ Example 5: Streaming Performance Measurement")
    print("-" * 50)

    client = StreamingVllmClient()

    print("📊 Measuring streaming performance...")
    print("👤 User: Write a detailed explanation of how neural networks work")

    import time

    try:
        start_time = time.time()
        first_token_time = None
        token_count = 0
        content_chunks = []

        async for event in client.stream_response(
            model="microsoft/DialoGPT-medium",
            input="Write a detailed explanation of how neural networks work",
            max_output_tokens=300
        ):
            current_time = time.time()

            if event["type"] == "content":
                if first_token_time is None:
                    first_token_time = current_time
                    print(f"⚡ Time to first token: {(first_token_time - start_time):.2f}s")
                    print("🤖 Response: ", end="", flush=True)

                content_delta = event.get("delta", "")
                print(content_delta, end="", flush=True)
                content_chunks.append(content_delta)
                token_count += 1

            elif event["type"] == "response.completed":
                end_time = current_time
                total_time = end_time - start_time
                streaming_time = end_time - (first_token_time or start_time)

                usage = event.get("usage", {})
                actual_tokens = usage.get("output_tokens", token_count)

                print(f"\n\n📈 Performance Metrics:")
                print(f"   Total time: {total_time:.2f}s")
                print(f"   Time to first token: {(first_token_time - start_time) if first_token_time else 0:.2f}s")
                print(f"   Streaming time: {streaming_time:.2f}s")
                print(f"   Tokens generated: {actual_tokens}")
                print(f"   Tokens per second: {actual_tokens / streaming_time if streaming_time > 0 else 0:.1f}")
                print(f"   Characters received: {len(''.join(content_chunks))}")

    except Exception as e:
        print(f"\n❌ Error during performance test: {e}")

    print("\n")


async def main():
    """Run all streaming examples."""
    print("🔄 vLLM Orchestrator - Streaming Examples")
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
    await example_1_basic_streaming()
    await example_2_streaming_with_tools()
    await example_3_streaming_conversation()
    await example_4_streaming_with_error_handling()
    await example_5_streaming_performance()

    print("🎉 All streaming examples completed!")
    print("\n📚 What you learned:")
    print("   ✅ How to use Server-Sent Events (SSE) for real-time responses")
    print("   ✅ How streaming works with tool calls")
    print("   ✅ How to maintain conversation context while streaming")
    print("   ✅ How to handle errors in streaming responses")
    print("   ✅ How to measure streaming performance")
    print("\n🚀 Next steps:")
    print("   - Try implementing your own streaming client")
    print("   - Experiment with different models and parameters")
    print("   - Build a real-time chat interface using these patterns")


if __name__ == "__main__":
    asyncio.run(main())