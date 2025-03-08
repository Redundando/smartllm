from smartllm import SmartLLM
import time
import os
import config

def chunk_callback(chunk):
    print(chunk, end="", flush=True)

api_key = config.ANTHROPIC_API_KEY
if not api_key:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    exit(1)

print("Creating SmartLLM instance with streaming enabled...")
llm = SmartLLM(
    base="anthropic",
    model="claude-3-7-sonnet-20250219",
    api_key=api_key,
    prompt="Write a short poem about artificial intelligence.",
    temperature=0.7,
    max_output_tokens=1000,
    stream=True
)

print("\nStarting request with background streaming...")
llm.generate_streaming(callback=chunk_callback)

print("\n\nContinuing with other work while streaming happens in background...")

for i in range(5):
    print(f"Doing other work... ({i+1}/5)")
    time.sleep(1)

print("\nWaiting for streaming to complete...")
llm.wait_for_completion()

if llm.is_failed():
    print(f"\nError occurred: {llm.get_error()}")
else:
    print("\nStreaming completed successfully!")

print("\n--- Creating a second instance with the same parameters ---")
print("This should use the cached result (instant playback):")

llm2 = SmartLLM(
    base="anthropic",
    model="claude-3-7-sonnet-20250219",
    api_key=api_key,
    prompt="Write a short poem about artificial intelligence.",
    temperature=0.7,
    max_output_tokens=1000,
    stream=True
)

print("\nStarting second request (using cache):")
llm2.generate_streaming(callback=chunk_callback)