import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartllm.bedrock import BedrockLLMClient, BedrockConfig


async def main():
    async with BedrockLLMClient(BedrockConfig(aws_region="us-east-1")) as client:
        models = await client.list_available_model_ids()
        sonnet_models = [m for m in models if "sonnet" in m.lower()]
        print(sonnet_models)


asyncio.run(main())
