import os
import time
from smartllm import SmartLLM
from logorator import Logger


def main():
    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        return

    # Create 3 streaming instances with different prompts
    stream1 = SmartLLM(
            base="anthropic",
            model="claude-3-7-sonnet-20250219",
            api_key=api_key,
            prompt="Write a short paragraph about space exploration.",
            max_output_tokens=300,
            stream=True
    )

    stream2 = SmartLLM(
            base="anthropic",
            model="claude-3-7-sonnet-20250219",
            api_key=api_key,
            prompt="Write a short paragraph about underwater discoveries.",
            max_output_tokens=300,
            stream=True
    )

    stream3 = SmartLLM(
            base="anthropic",
            model="claude-3-7-sonnet-20250219",
            api_key=api_key,
            prompt="Write a short paragraph about rainforest conservation.",
            max_output_tokens=300,
            stream=True
    )

    # Start all streams (they will run concurrently)
    print("Starting stream 1...")
    stream1.stream()  # Using default callback

    print("Starting stream 2...")
    stream2.stream()  # Using default callback

    print("Starting stream 3...")
    stream3.stream()  # Using default callback

    # Check status periodically
    while not (stream1.is_completed() and stream2.is_completed() and stream3.is_completed()):
        # Check if any stream failed
        if stream1.is_failed() or stream2.is_failed() or stream3.is_failed():
            print("One of the streams failed!")
            break

        # Wait a bit before checking again
        time.sleep(0.5)
        print(".", end="", flush=True)

    print("\nAll streams completed!")

    # Print results
    print("\n--- Stream 1 Result ---")
    print(f"Content length: {len(stream1.content)}")
    print(stream1.content[:100] + "...")

    print("\n--- Stream 2 Result ---")
    print(f"Content length: {len(stream2.content)}")
    print(stream2.content[:100] + "...")

    print("\n--- Stream 3 Result ---")
    print(f"Content length: {len(stream3.content)}")
    print(stream3.content[:100] + "...")


if __name__ == "__main__":
    main()