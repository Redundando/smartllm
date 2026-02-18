"""Example: Reasoning with structured output using OpenAI Response API"""

import asyncio
import sys
from pathlib import Path
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartllm import LLMClient, TextRequest


class MathSolution(BaseModel):
    """Solution to a math problem"""
    answer: float
    unit: str
    explanation: str


async def main():
    async with LLMClient(provider="openai") as client:
        response = await client.generate_text(
            TextRequest(
                prompt="A train leaves city A at 60mph. Another leaves city B (300 miles away) at 90mph traveling toward each other. When and where do they meet?",
                model="gpt-5.2",
                reasoning_effort="high",
                response_format=MathSolution,
                max_tokens=500,
                clear_cache=True,
            )
        )

    solution = response.structured_data
    print(f"Answer:      {solution.answer} {solution.unit}")
    print(f"Explanation: {solution.explanation}")
    print(f"\nTokens: {response.input_tokens} in / {response.output_tokens} out")
    print(f"Reasoning tokens: {response.metadata.get('reasoning_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
