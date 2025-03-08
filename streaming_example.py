from smartllm import SmartLLM
import os
import sys
import config

def handle_chunk(chunk):
    """Process each chunk as it arrives."""
    # Print the chunk without a newline and flush the buffer
    print(chunk, end="", flush=True)

    # You can also process the chunk in other ways:
    # - Update a UI
    # - Accumulate specific information
    # - Parse for special tokens/commands
    # - etc.


def main():
    # Get API key from environment variable
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Create an instance of SmartLLM with streaming enabled
    llm = SmartLLM(
        base="anthropic",
        model="claude-3-7-sonnet-20250219",
        api_key=api_key,
        prompt="Write a short story about a robot discovering emotions. Make it unfold gradually.",
        temperature=0.7,
        max_output_tokens=1000,
        stream=True  # This enables streaming mode
    )

    print("Generating response...\n")

    # Start generating with streaming
    llm.generate_streaming(callback=handle_chunk)

    # Check for errors after completion
    if llm.is_failed():
        print(f"\n\nError occurred: {llm.get_error()}")
        return

    print("\n\nGeneration complete!")


if __name__ == "__main__":
    main()