"""Load testing script for vLLM Orchestrator."""

import asyncio
import json
import time
from typing import Dict, List

import httpx

from vllm_orchestrator.observability.logging import get_logger

logger = get_logger(__name__)


class LoadTester:
    """Load testing utility for vLLM Orchestrator."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = None,
        max_concurrent: int = 10,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Test statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
        }

    async def run_load_test(
        self,
        test_scenarios: List[Dict],
        duration_seconds: int = 60,
        requests_per_second: int = 5,
    ) -> Dict:
        """Run load test with given scenarios.

        Args:
            test_scenarios: List of test scenario configurations
            duration_seconds: How long to run the test
            requests_per_second: Target RPS

        Returns:
            Test statistics
        """
        logger.info(
            "Starting load test",
            duration=duration_seconds,
            rps=requests_per_second,
            scenarios=len(test_scenarios),
        )

        start_time = time.time()
        end_time = start_time + duration_seconds

        tasks = []

        while time.time() < end_time:
            # Create batch of requests
            for scenario in test_scenarios:
                if time.time() >= end_time:
                    break

                task = asyncio.create_task(self._execute_scenario(scenario))
                tasks.append(task)

                # Rate limiting
                await asyncio.sleep(1.0 / requests_per_second / len(test_scenarios))

        # Wait for all tasks to complete
        logger.info("Waiting for requests to complete", pending_tasks=len(tasks))
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate final statistics
        return self._calculate_stats()

    async def _execute_scenario(self, scenario: Dict) -> None:
        """Execute a single test scenario."""
        async with self.semaphore:
            start_time = time.time()

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["x-api-key"] = self.api_key

                    url = f"{self.base_url}{scenario['endpoint']}"
                    method = scenario.get("method", "POST")

                    if method == "POST":
                        response = await client.post(
                            url, json=scenario["data"], headers=headers
                        )
                    elif method == "GET":
                        response = await client.get(url, headers=headers)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    response_time = time.time() - start_time

                    self.stats["total_requests"] += 1
                    self.stats["response_times"].append(response_time)

                    if 200 <= response.status_code < 300:
                        self.stats["successful_requests"] += 1
                        logger.debug(
                            "Request successful",
                            endpoint=scenario["endpoint"],
                            status_code=response.status_code,
                            response_time=response_time,
                        )
                    else:
                        self.stats["failed_requests"] += 1
                        error_info = {
                            "endpoint": scenario["endpoint"],
                            "status_code": response.status_code,
                            "response": response.text[:200],
                            "response_time": response_time,
                        }
                        self.stats["errors"].append(error_info)
                        logger.warning("Request failed", **error_info)

            except Exception as e:
                response_time = time.time() - start_time
                self.stats["total_requests"] += 1
                self.stats["failed_requests"] += 1
                self.stats["response_times"].append(response_time)

                error_info = {
                    "endpoint": scenario["endpoint"],
                    "error": str(e),
                    "response_time": response_time,
                }
                self.stats["errors"].append(error_info)
                logger.error("Request exception", **error_info)

    def _calculate_stats(self) -> Dict:
        """Calculate final test statistics."""
        if not self.stats["response_times"]:
            return self.stats

        response_times = self.stats["response_times"]
        response_times.sort()

        n = len(response_times)
        stats = {
            **self.stats,
            "success_rate": self.stats["successful_requests"]
            / max(self.stats["total_requests"], 1),
            "avg_response_time": sum(response_times) / n,
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p50_response_time": response_times[n // 2],
            "p90_response_time": response_times[int(n * 0.9)],
            "p95_response_time": response_times[int(n * 0.95)],
            "p99_response_time": response_times[int(n * 0.99)],
        }

        return stats


def create_test_scenarios() -> List[Dict]:
    """Create test scenarios for load testing."""
    scenarios = [
        # Health check
        {"name": "health_check", "endpoint": "/health", "method": "GET", "data": None},
        # Simple response generation
        {
            "name": "simple_response",
            "endpoint": "/v1/responses",
            "method": "POST",
            "data": {
                "model": "test-model",
                "input": "Hello, how are you?",
                "max_output_tokens": 100,
            },
        },
        # Response with file search
        {
            "name": "response_with_file_search",
            "endpoint": "/v1/responses",
            "method": "POST",
            "data": {
                "model": "test-model",
                "input": "Find information about machine learning",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "file_search",
                            "description": "Search files",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "vector_store_id": {"type": "string"},
                                },
                                "required": ["query", "vector_store_id"],
                            },
                        },
                    }
                ],
                "max_output_tokens": 200,
            },
        },
        # Vector store search
        {
            "name": "vector_search",
            "endpoint": "/v1/vector_stores/vs_test123/search",
            "method": "POST",
            "data": {"query": "machine learning algorithms", "max_results": 5},
        },
        # File upload
        {
            "name": "file_upload",
            "endpoint": "/v1/files",
            "method": "POST",
            "data": {"filename": "test.txt", "purpose": "assistants"},
        },
    ]

    return scenarios


async def main():
    """Run the load test."""
    import argparse

    parser = argparse.ArgumentParser(description="Load test vLLM Orchestrator")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument("--rps", type=int, default=5, help="Requests per second")
    parser.add_argument(
        "--concurrent", type=int, default=10, help="Max concurrent requests"
    )

    args = parser.parse_args()

    # Create load tester
    tester = LoadTester(
        base_url=args.url, api_key=args.api_key, max_concurrent=args.concurrent
    )

    # Create test scenarios
    scenarios = create_test_scenarios()

    # Run load test
    stats = await tester.run_load_test(
        test_scenarios=scenarios,
        duration_seconds=args.duration,
        requests_per_second=args.rps,
    )

    # Print results
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful_requests']}")
    print(f"Failed: {stats['failed_requests']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print()
    print("Response Times (seconds):")
    print(f"  Average: {stats['avg_response_time']:.3f}")
    print(f"  Min: {stats['min_response_time']:.3f}")
    print(f"  Max: {stats['max_response_time']:.3f}")
    print(f"  P50: {stats['p50_response_time']:.3f}")
    print(f"  P90: {stats['p90_response_time']:.3f}")
    print(f"  P95: {stats['p95_response_time']:.3f}")
    print(f"  P99: {stats['p99_response_time']:.3f}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"  {error}")

    print("=" * 60)

    # Save detailed results
    with open("load_test_results.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Detailed results saved to load_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
