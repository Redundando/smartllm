from smartllm import SmartLLM
import os
import sys
import config

def handle_chunk(chunk):
    """Process each chunk as it arrives."""
    # Print the chunk without a newline and flush the buffer
    print(chunk, end="", flush=True)

def main():
    # Get API key from environment variable
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    print("Example 1: Basic streaming with custom callback")
    print("-----------------------------------------------")

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

    print("Generating response with custom callback...\n")

    # Start generating with streaming
    llm.generate_streaming(callback=handle_chunk)

    # Check for errors after completion
    if llm.is_failed():
        print(f"\n\nError occurred: {llm.get_error()}")
    else:
        print("\n\nGeneration complete!")

    print("\n\nExample 2: Streaming with default logging callback")
    print("--------------------------------------------------")

    # Create another instance with streaming enabled
    llm2 = SmartLLM(
        base="anthropic",
        model="claude-3-7-sonnet-20250219",
        api_key=api_key,
        prompt="Explain quantum computing in simple terms.",
        temperature=0.7,
        max_output_tokens=1000,
        stream=True
    )

    print("Generating response with default callback...\n")

    # Start generating with streaming, using the default callback
    llm2.generate_streaming()

    # Check for errors after completion
    if llm2.is_failed():
        print(f"Error occurred: {llm2.get_error()}")
    else:
        print("Generation complete!")
        print("\nFinal response:")
        print(llm2.content)

if __name__ == "__main__":
    main()