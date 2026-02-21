import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartllm import LLMClient, TextRequest


async def main():
    async with LLMClient(provider="bedrock") as client:
        response = await client.generate_text(
            TextRequest(
                prompt="What is the capital of France?",
                model="us.anthropic.claude-sonnet-4-6",
            )
        )
        print(response.text)


asyncio.run(main())
