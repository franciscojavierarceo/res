#!/usr/bin/env python3
"""File search (RAG) example for vLLM Orchestrator.

This script demonstrates:
- File upload functionality
- Vector store creation and management
- File search tool usage in responses
- End-to-end RAG workflow
"""

import asyncio
import tempfile
import httpx
from typing import Optional
from pathlib import Path


class VllmOrchestratorClient:
    """Client for vLLM Orchestrator with file and vector store support."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient()
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    async def upload_file(self, file_path: str, purpose: str = "assistants") -> dict:
        """Upload a file."""
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            data = {"purpose": purpose}
            response = await self.client.post(
                f"{self.base_url}/v1/files",
                files=files,
                data=data,
                headers={k: v for k, v in self.headers.items() if k != "Content-Type"}
            )
        response.raise_for_status()
        return response.json()

    async def create_vector_store(self, name: str, file_ids: Optional[list] = None) -> dict:
        """Create a vector store."""
        payload = {"name": name}
        if file_ids:
            payload["file_ids"] = file_ids

        response = await self.client.post(
            f"{self.base_url}/v1/vector_stores",
            json=payload,
            headers={**self.headers, "Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    async def add_file_to_vector_store(self, vector_store_id: str, file_id: str) -> dict:
        """Add a file to a vector store."""
        response = await self.client.post(
            f"{self.base_url}/v1/vector_stores/{vector_store_id}/files",
            json={"file_id": file_id},
            headers={**self.headers, "Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    async def search_vector_store(self, vector_store_id: str, query: str, max_results: int = 5) -> dict:
        """Search a vector store."""
        response = await self.client.post(
            f"{self.base_url}/v1/vector_stores/{vector_store_id}/search",
            json={"query": query, "max_results": max_results},
            headers={**self.headers, "Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    async def create_response_with_file_search(
        self,
        model: str,
        input: str,
        vector_store_ids: list,
        max_output_tokens: int = 200
    ) -> dict:
        """Create a response with file search tool."""
        payload = {
            "model": model,
            "input": input,
            "tools": [{
                "type": "file_search",
                "file_search": {
                    "vector_store_ids": vector_store_ids
                }
            }],
            "max_output_tokens": max_output_tokens
        }

        response = await self.client.post(
            f"{self.base_url}/v1/responses",
            json=payload,
            headers={**self.headers, "Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = {
        "machine_learning_basics.txt": """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence (AI) that focuses on developing algorithms and statistical models that enable computers to improve their performance on a specific task through experience, without being explicitly programmed.

Key Types of Machine Learning:
1. Supervised Learning: Uses labeled data to train models
2. Unsupervised Learning: Finds patterns in unlabeled data
3. Reinforcement Learning: Learns through interaction and feedback

Popular Algorithms:
- Linear Regression: For predicting continuous values
- Decision Trees: For classification and regression
- Neural Networks: For complex pattern recognition
- K-Means: For clustering similar data points

Applications:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Predictive analytics
- Autonomous vehicles
""",
        "python_programming.txt": """
Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation.

Key Features:
- Simple and readable syntax
- Dynamically typed
- Extensive standard library
- Large ecosystem of third-party packages
- Cross-platform compatibility

Popular Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Scikit-learn: Machine learning
- TensorFlow/PyTorch: Deep learning
- Django/Flask: Web development

Best Practices:
1. Follow PEP 8 style guidelines
2. Use virtual environments
3. Write comprehensive tests
4. Document your code
5. Use type hints for better code quality

Common Use Cases:
- Data analysis and visualization
- Web application development
- Machine learning and AI
- Automation and scripting
- Scientific computing
""",
        "transformers_architecture.txt": """
Transformer Architecture in Deep Learning

The Transformer architecture, introduced in the paper "Attention Is All You Need" (2017), revolutionized natural language processing and has become the foundation for modern language models.

Key Components:

1. Self-Attention Mechanism:
   - Allows the model to focus on different parts of the input sequence
   - Computes attention weights for each token relative to all other tokens
   - Enables parallel processing unlike RNNs

2. Multi-Head Attention:
   - Uses multiple attention heads to capture different types of relationships
   - Each head learns different patterns and representations
   - Outputs are concatenated and linearly transformed

3. Position Encoding:
   - Adds positional information since transformers don't have inherent sequence order
   - Uses sinusoidal functions or learned embeddings
   - Critical for understanding word order and sentence structure

4. Feed-Forward Networks:
   - Applied to each position independently
   - Consists of two linear layers with ReLU activation
   - Provides non-linear transformation capabilities

Advantages:
- Parallel computation during training
- Better handling of long-range dependencies
- More efficient than RNNs for long sequences
- Excellent transfer learning capabilities

Applications:
- BERT, GPT, T5 language models
- Machine translation systems
- Text summarization
- Question answering systems
- Code generation
"""
    }

    temp_files = []
    for filename, content in documents.items():
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f"_{filename}", delete=False)
        temp_file.write(content)
        temp_file.flush()
        temp_files.append(temp_file.name)

    return temp_files


async def example_1_file_upload_and_vector_store():
    """Example 1: Upload files and create vector store."""
    print("📁 Example 1: File Upload and Vector Store Creation")
    print("-" * 60)

    client = VllmOrchestratorClient()

    try:
        # Create sample documents
        temp_files = create_sample_documents()
        print(f"📄 Created {len(temp_files)} sample documents")

        # Upload files
        uploaded_files = []
        for file_path in temp_files:
            filename = Path(file_path).name
            print(f"⬆️  Uploading {filename}...")

            file_response = await client.upload_file(file_path, purpose="assistants")
            uploaded_files.append(file_response)

            print(f"   ✅ File ID: {file_response['id']}")
            print(f"   📊 Size: {file_response['bytes']} bytes")

        print()

        # Create vector store
        print("🏗️  Creating vector store...")
        file_ids = [f["id"] for f in uploaded_files]
        vector_store = await client.create_vector_store(
            name="ml_programming_docs",
            file_ids=file_ids
        )

        print(f"✅ Vector store created: {vector_store['id']}")
        print(f"📚 Files added: {len(vector_store.get('file_counts', {}).get('total', 0))}")

        # Clean up temp files
        for file_path in temp_files:
            Path(file_path).unlink()

        return vector_store["id"]

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    finally:
        await client.close()


async def example_2_vector_search():
    """Example 2: Direct vector search."""
    print("🔍 Example 2: Direct Vector Search")
    print("-" * 60)

    # First create vector store
    vector_store_id = await example_1_file_upload_and_vector_store()
    if not vector_store_id:
        print("❌ Could not create vector store, skipping search example")
        return None

    print()

    client = VllmOrchestratorClient()

    try:
        search_queries = [
            "What are the key components of transformer architecture?",
            "How do I get started with machine learning?",
            "What Python libraries are good for data science?"
        ]

        for i, query in enumerate(search_queries, 1):
            print(f"🔍 Search {i}: {query}")

            results = await client.search_vector_store(
                vector_store_id=vector_store_id,
                query=query,
                max_results=3
            )

            print(f"📊 Found {len(results.get('data', []))} relevant chunks:")

            for j, result in enumerate(results.get("data", []), 1):
                score = result.get("score", 0)
                content_preview = result.get("content", "")[:150] + "..."
                print(f"   {j}. Score: {score:.3f}")
                print(f"      Content: {content_preview}")
                print()

            print()

        return vector_store_id

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    finally:
        await client.close()


async def example_3_file_search_in_responses():
    """Example 3: Using file search in response generation."""
    print("🤖 Example 3: File Search in Response Generation")
    print("-" * 60)

    # Get vector store from previous example
    vector_store_id = await example_2_vector_search()
    if not vector_store_id:
        print("❌ Could not get vector store, skipping response example")
        return

    print()

    client = VllmOrchestratorClient()

    try:
        questions = [
            "What is self-attention in transformers and why is it important?",
            "I'm new to machine learning. What should I learn first?",
            "What Python libraries would you recommend for someone getting started with data science and machine learning?",
            "How do transformers differ from RNNs in terms of processing sequences?"
        ]

        for i, question in enumerate(questions, 1):
            print(f"❓ Question {i}: {question}")

            response = await client.create_response_with_file_search(
                model="microsoft/DialoGPT-medium",
                input=question,
                vector_store_ids=[vector_store_id],
                max_output_tokens=250
            )

            print(f"🤖 Response: {response['output']['content']}")

            # Show token usage
            usage = response.get('usage', {})
            print(f"📊 Tokens: {usage.get('input_tokens', 0)} input + {usage.get('output_tokens', 0)} output = {usage.get('total_tokens', 0)} total")

            print()

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await client.close()


async def example_4_rag_conversation():
    """Example 4: Multi-turn RAG conversation."""
    print("💬 Example 4: Multi-turn RAG Conversation")
    print("-" * 60)

    # Use vector store from previous examples
    vector_store_id = await example_1_file_upload_and_vector_store()
    if not vector_store_id:
        print("❌ Could not create vector store, skipping conversation example")
        return

    print()

    client = VllmOrchestratorClient()

    try:
        conversation = [
            "I want to learn machine learning. What are the main types I should know about?",
            "Which one would be best for analyzing customer data to predict purchases?",
            "What Python libraries would I need for that type of analysis?",
            "How do transformers fit into machine learning? Are they related to supervised learning?"
        ]

        previous_response_id = None

        for i, question in enumerate(conversation, 1):
            print(f"👤 User: {question}")

            # Create response with file search
            payload = {
                "model": "microsoft/DialoGPT-medium",
                "input": question,
                "tools": [{
                    "type": "file_search",
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }],
                "max_output_tokens": 200
            }

            if previous_response_id:
                payload["previous_response_id"] = previous_response_id

            response = await client.client.post(
                f"{client.base_url}/v1/responses",
                json=payload,
                headers={**client.headers, "Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_data = response.json()

            print(f"🤖 Assistant: {response_data['output']['content']}")

            # Update for next iteration
            previous_response_id = response_data["id"]

            print()

        print("🎯 Conversation completed! The assistant used file search to provide")
        print("   contextual answers while maintaining conversation history.")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await client.close()


async def main():
    """Run all file search examples."""
    print("🔍 vLLM Orchestrator - File Search (RAG) Examples")
    print("=" * 70)
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
    await example_1_file_upload_and_vector_store()
    print()
    await example_2_vector_search()
    print()
    await example_3_file_search_in_responses()
    print()
    await example_4_rag_conversation()

    print("🎉 All file search examples completed!")
    print("\n📚 What you learned:")
    print("   ✅ How to upload files to vLLM Orchestrator")
    print("   ✅ How to create and manage vector stores")
    print("   ✅ How to perform semantic search on documents")
    print("   ✅ How to use file_search tool in responses")
    print("   ✅ How to maintain conversation context with RAG")
    print("\n🚀 Next steps:")
    print("   - Try the streaming example: python examples/streaming_example.py")
    print("   - Upload your own documents and ask questions about them")
    print("   - Experiment with different search queries and parameters")


if __name__ == "__main__":
    asyncio.run(main())