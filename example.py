from smartllm import SmartLLM
from smartllm import AsyncSmartLLM
import os
import asyncio



async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    json_schema = {"type": "object", "properties": {"message": {"type": "string", "description": "A simple text message"}}, "required": ["message"]}

    # Create and execute in one step
    llm = AsyncSmartLLM(base="anthropic", model="claude-3-7-sonnet-20250219", api_key=api_key, prompt="Give me a short, uplifting message, please", temperature=0.7,
            max_output_tokens=1000, stream = True)
    await llm.execute()
    print(llm.response)


    #if llm.is_failed():
    #    print(f"Error occurred: {llm.get_error()}")
    #else:
    #    print("\nResponse received:\n")
    #    print(llm.response)

    # Streaming example
    """
    def print_chunk(chunk: str, accumulated: str) -> None:
        print(chunk, end="", flush=True)

    print("\n\nStreaming example:\n")
    streaming_llm = SmartLLM(base="anthropic", model="claude-3-7-sonnet-20250219", api_key=api_key, prompt="Tell me a short joke about programming", stream=True).execute(callback=print_chunk)

    print("\n\nFinal response:")
    print(streaming_llm.response)
    """

if __name__ == "__main__":
    asyncio.run(main())