"""Example: Caching - same prompt is served from cache on second call"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartllm import LLMClient, TextRequest

PROMPT = "What is the capital of France?"


async def main():
    async with LLMClient(provider="openai") as client:
        # First call - hits the API
        t0 = time.perf_counter()
        r1 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0))
        t1 = time.perf_counter()
        print(f"First call  ({t1 - t0:.2f}s): {r1.text}")

        # Second call - served from cache
        t0 = time.perf_counter()
        r2 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0))
        t1 = time.perf_counter()
        print(f"Second call ({t1 - t0:.2f}s): {r2.text}")

        # Force fresh API call by clearing cache
        t0 = time.perf_counter()
        r3 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0, clear_cache=True))
        t1 = time.perf_counter()
        print(f"Cache clear ({t1 - t0:.2f}s): {r3.text}")


if __name__ == "__main__":
    asyncio.run(main())
