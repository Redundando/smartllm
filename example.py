from smartllm import SmartLLM
import time
import os
import config

def main():
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    print("Creating SmartLLM instance (non-streaming)...")
    llm = SmartLLM(
        base="anthropic",
        model="claude-3-7-sonnet-20250219",
        api_key=api_key,
        prompt="What are the three most important considerations when designing a REST API?",
        temperature=0.7,
        max_output_tokens=1000
    )

    print("Starting non-streaming request...")
    llm.generate()

    print("Request running in background...")
    print("Doing other work while waiting for response...")

    # Simulate doing other work while request runs
    for i in range(3):
        print(f"Working... ({i + 1}/3)")
        time.sleep(1)

    print("Waiting for completion...")
    llm.wait_for_completion()

    if llm.is_failed():
        print(f"Error occurred: {llm.get_error()}")
    else:
        print("\nResponse received:\n")
        print(llm.content)

    print("\n--- Creating a second instance with the same parameters ---")
    print("This should use the cached result:")

    llm2 = SmartLLM(
        base="anthropic",
        model="claude-3-7-sonnet-20250219",
        api_key=api_key,
        prompt="What are the three most important considerations when designing a REST API?",
        temperature=0.7,
        max_output_tokens=1000
    )

    print("Generating response from cache...")
    llm2.generate().wait_for_completion()

    print("Response from cache:")
    print(llm2.content)


if __name__ == "__main__":
    main()