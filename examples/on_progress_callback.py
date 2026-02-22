import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smartllm import LLMClient, TextRequest


async def main():
    async with LLMClient(provider="openai") as client:
        # First call — clears cache to force API hit, fires llm_started + llm_done
        print("--- First call (API) ---")
        await client.generate_text(TextRequest(
            prompt="What is the capital of France?",
            temperature=0,
            clear_cache=True,
            on_progress=print,
        ))

        # Second call — served from cache, fires llm_started + cache_hit
        print("\n--- Second call (cache) ---")
        await client.generate_text(TextRequest(
            prompt="What is the capital of France?",
            temperature=0,
            on_progress=print,
        ))


asyncio.run(main())
