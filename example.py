import asyncio
import os

from smartllm import AsyncSmartLLM


async def print_chunk(chunk: str, accumulated: str) -> None:
    print(chunk, end="", flush=True)


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    json_schema = {"type": "object", "properties": {"story": {"type": "string", "description": "A simple text message"}}, "required": ["story"]}

    # Streaming example with async callback
    print("\nStreaming example:\n")
    llm = AsyncSmartLLM(base="anthropic", model="claude-3-7-sonnet-20250219", api_key=api_key, prompt="Give me a funny story, please", temperature=0.7, max_output_tokens=1000, stream=True, json_mode=True,
            json_schema=json_schema)

    # Define a synchronous callback function that will be called from async code
    def sync_print_chunk(chunk: str, accumulated: str) -> None:
        print(chunk, end="", flush=True)

    await llm.execute(callback=sync_print_chunk)

    if llm.is_failed():
        print(f"\nError occurred: {llm.get_error()}")
    else:
        print("\n\nFinal response:")
        print(llm.response)


if __name__ == "__main__":
    asyncio.run(main())
