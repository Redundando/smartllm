import asyncio
from pydantic import BaseModel
from smartllm import LLMClient, TextRequest


class Solution(BaseModel):
    answer: float
    unit: str
    explanation: str


async def main():
    async with LLMClient(provider="openai") as client:
        response = await client.generate_text(
            TextRequest(
                prompt="A snail travels at 0.03 mph. A cheetah runs at 75 mph. If the snail gets a 10-mile head start, how long before the cheetah catches up?",
                response_format=Solution,
                model="gpt-5.2",
                reasoning_effort="high",
                clear_cache=True,
            )
        )

    print(response)


asyncio.run(main())
