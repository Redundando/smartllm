import asyncio
import os
from smartllm import AsyncSmartLLM


async def process_number(i: int, api_key: str) -> dict:
    # Create JSON schema for the response
    json_schema = {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}, "description": "Interesting facts about the number"},
            "most_interesting_fact"                        : {"type": "string", "description": "The single most interesting fact"}}, "required": ["facts", "most_interesting_fact"]}

    # Create an LLM instance with streaming and JSON mode
    llm = AsyncSmartLLM(base="anthropic", model="claude-3-7-sonnet-20250219", api_key=api_key, prompt=f"Tell me the most interesting facts about the number {i}", temperature=0.7, max_output_tokens=1000, stream=True,
            json_mode=True, json_schema=json_schema)

    # Execute the request
    await llm.execute()

    # Return the response
    return {"number": i, "success": llm.is_completed(), "response": llm.response if llm.is_completed() else llm.get_error()}


async def main():
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    print("Starting Many concurrent requests...")

    # Create and gather all tasks
    tasks = [process_number(i, api_key) for i in range(1, 100)]
    results = await asyncio.gather(*tasks)

    # Print results
    for result in results:
        number = result["number"]
        if result["success"]:
            print(f"\nNumber {number}: {result['response']['most_interesting_fact']}")
        else:
            print(f"\nNumber {number} error: {result['response']}")


if __name__ == "__main__":
    asyncio.run(main())