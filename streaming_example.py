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
llm.generate_response(callback=chunk_callback)
