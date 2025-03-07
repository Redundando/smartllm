"""
Example showing how to use the SmartLLM sources functionality
"""
from smartllm.smart_llm import SmartLLM
import _config as config

def perplexity_sources_example():
    """
    Example using Perplexity's citation capabilities
    """
    # Create SmartLLM instance with Perplexity (which provides citations)
    llm = SmartLLM(
            base="perplexity",
            model="sonar-pro",
            api_key=config.PERPLEXITY_API_KEY,
            prompt="Ich schreibe einen Artikel über das Buch 'Demian' von Hermann Hesse. Liste bitte alle relevanten Quellen zu diesem Buch auf: Inhalt, Rezeption, Details, etc.",
    )

    # Generate response and wait for completion
    llm.generate().wait_for_completion()

    if llm.is_failed():
        print(f"ERROR: {llm.get_error()}")
        return

    # Display the response
    print("RESPONSE FROM PERPLEXITY:")
    print(llm.content)
    print("\nSOURCES:")

    # Access the sources
    if llm.sources:
        for i, source in enumerate(llm.sources, 1):
            print(f"{i}. {source}")
    else:
        print("No sources provided")


def anthropic_sources_example():
    """
    Example showing sources property works with Anthropic too, even though
    Anthropic doesn't typically provide formal citations
    """
    llm = SmartLLM(
            base="anthropic",
            model="claude-3-7-sonnet-20250219",  # Use appropriate model
            api_key=config.ANTHROPIC_API_KEY,
            prompt="What is the tallest building in the world as of 2024?",
            temperature=0.0  # Low temperature for factual information
    )

    # Generate response and wait for completion
    llm.generate().wait_for_completion()

    if llm.is_failed():
        print(f"ERROR: {llm.get_error()}")
        return

    # Display the response
    print("\n\nRESPONSE FROM ANTHROPIC:")
    print(llm.content)
    print("\nSOURCES:")

    # Access the sources (likely empty for Anthropic)
    if llm.sources:
        for i, source in enumerate(llm.sources, 1):
            print(f"{i}. {source}")
    else:
        print("No sources provided (as expected with Anthropic)")


def openai_sources_example():
    """
    Example showing sources property works with OpenAI too
    """
    llm = SmartLLM(
            base="openai",
            model="gpt-4o",  # Use appropriate model
            api_key=config.OPENAI_API_KEY,
            prompt="What is the tallest building in the world as of 2024?",
            temperature=0.0  # Low temperature for factual information
    )

    # Generate response and wait for completion
    llm.generate().wait_for_completion()

    if llm.is_failed():
        print(f"ERROR: {llm.get_error()}")
        return

    # Display the response
    print("\n\nRESPONSE FROM OPENAI:")
    print(llm.content)
    print("\nSOURCES:")

    # Access the sources (might be empty for OpenAI depending on model)
    if llm.sources:
        for i, source in enumerate(llm.sources, 1):
            print(f"{i}. {source}")
    else:
        print("No sources provided")


if __name__ == "__main__":
    # Run the examples
    perplexity_sources_example()
    #anthropic_sources_example()
    #openai_sources_example()