import os
from smartllm import SmartLLM


def print_chunk(chunk: str, accumulated: str) -> None:
    print(f"CHUNK: {chunk}")


def main():
    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    api_key = os.environ.get("PERPLEXITY_API_KEY")

    # JSON schema for a developer profile
    json_schema = {
            "type"      : "object",
            "properties": {
                    "name"      : {"type": "string"},
                    "skills"    : {"type": "array", "items": {"type": "string"}},
                    "experience": {"type": "integer"}
            }
    }

    # Create SmartLLM instance with both streaming and JSON mode
    llm = SmartLLM(
            base="perplexity",
            model="sonar-pro",
            api_key=api_key,
            prompt="Create a short profile for a senior developer",
            system_prompt="You are a joker and always and only make funny jokes.",
            #stream=True,
            json_mode=True,
            json_schema=json_schema,
            #clear_cache=True
    )

    # Start streaming
    llm.execute()

    # Wait for completion
    llm.wait_for_completion()

    # Print final result
    print("\nFINAL RESULT:")
    print(llm.content)


if __name__ == "__main__":
    main()