"""Example: DynamoDB two-level caching - local file + DynamoDB shared cache

Requires:
    pip install dynamorator
    AWS credentials with DynamoDB access

Set DYNAMO_TABLE env var or edit TABLE_NAME below.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartllm import LLMClient, TextRequest

TABLE_NAME = os.getenv("DYNAMO_TABLE", "smartllm-cache")
PROMPT = "What is the capital of France?"


async def main():
    async with LLMClient(provider="openai", dynamo_table_name=TABLE_NAME) as client:
        # First call - misses both caches, hits API, writes to local + DynamoDB
        t0 = time.perf_counter()
        r1 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0, clear_cache=True))
        print(f"API call    ({time.perf_counter() - t0:.2f}s): {r1.text}")

        # Second call - local cache hit (instant)
        t0 = time.perf_counter()
        r2 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0))
        print(f"Local hit   ({time.perf_counter() - t0:.2f}s): {r2.text}")

        # Simulate cold local cache by clearing local files only
        client._client.cache.local.clear()
        print("\nLocal cache cleared (simulating fresh machine)...")

        # Third call - local miss, DynamoDB hit, writes back to local
        t0 = time.perf_counter()
        r3 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0))
        print(f"DB hit      ({time.perf_counter() - t0:.2f}s): {r3.text}")

        # Fourth call - local cache warm again from DB write-back
        t0 = time.perf_counter()
        r4 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0))
        print(f"Local hit   ({time.perf_counter() - t0:.2f}s): {r4.text}")


if __name__ == "__main__":
    asyncio.run(main())
