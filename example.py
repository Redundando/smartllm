from smartllm import SmartLLM
import os

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    api_key = os.environ.get("PERPLEXITY_API_KEY")

    print("Creating SmartLLM instance (non-streaming)...")
    llm = SmartLLM(
        base="perplexity",
        model="sonar-pro",
        api_key=api_key,
        prompt="What are the three most important considerations when designing a REST API?",
        temperature=0.7,
        max_output_tokens=1000
    )

    print("Starting non-streaming request...")
    llm.generate()
    print("Waiting for completion...")
    llm.wait_for_completion()

    if llm.is_failed():
        print(f"Error occurred: {llm.get_error()}")
    else:
        print("\nResponse received:\n")
        print(llm.content)
        print(llm.sources)


if __name__ == "__main__":
    main()