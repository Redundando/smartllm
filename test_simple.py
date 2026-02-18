"""Simple test script for SmartLLM with OpenAI"""

import asyncio
from smartllm import LLMClient, TextRequest


async def main():
    """Test text generation with OpenAI"""
    
    async with LLMClient(provider="openai") as client:
        # Simple text generation
        response = await client.generate_text(
            TextRequest(
                prompt="Solve step by step: A train leaves city A at 60mph. Another train leaves city B (300 miles away) at 90mph. When do they meet?",
                model="gpt-5.2",
                reasoning_effort="medium",
                max_tokens=500,
                clear_cache=True
            )
        )
        
        print("Generated text:")
        print(response.text)
        print(f"\nTokens used: {response.input_tokens} in / {response.output_tokens} out")
        print(f"Metadata: {response.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
