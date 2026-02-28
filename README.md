# SmartLLM

A unified async Python wrapper for multiple LLM providers with a consistent interface.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Unified Interface** — Single API for OpenAI and AWS Bedrock
- **Async/Await** — Built on asyncio for concurrent requests
- **Smart Caching** — Two-level cache (local JSON + optional DynamoDB)
- **Auto Retry** — Exponential backoff for transient failures
- **Structured Output** — Native Pydantic model support
- **Streaming** — Real-time streaming responses
- **Rate Limiting** — Built-in concurrency control per model
- **Reasoning Models** — Full support including `reasoning_effort` and `reasoning_tokens`
- **Progress Callbacks** — Optional `on_progress` for real-time events

## Installation

```bash
pip install smartllm[openai]   # OpenAI only
pip install smartllm[bedrock]  # AWS Bedrock only
pip install smartllm[all]      # All providers
```

## Quick Start

```python
import asyncio
from smartllm import LLMClient, TextRequest

async def main():
    async with LLMClient(provider="openai") as client:
        response = await client.generate_text(
            TextRequest(prompt="What is the capital of France?")
        )
        print(response.text)

asyncio.run(main())
```

## Configuration

### Environment Variables

**OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # optional
```

**AWS Bedrock:**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
export BEDROCK_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"  # optional
```

Explicit credentials are optional. If omitted, boto3's default credential chain is used — including EC2 instance profiles, ECS task roles, Lambda execution roles, and `~/.aws/credentials`.

### Programmatic Configuration

```python
from smartllm import LLMClient, LLMConfig

config = LLMConfig(
    provider="openai",
    api_key="your-api-key",
    default_model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    max_retries=3,
)

async with LLMClient(config) as client:
    ...
```

## Usage Examples

### Multi-turn Conversations

```python
from smartllm import LLMClient, MessageRequest, Message

async with LLMClient(provider="openai") as client:
    messages = [
        Message(role="user", content="My name is Alice."),
        Message(role="assistant", content="Nice to meet you, Alice!"),
        Message(role="user", content="What's my name?"),
    ]
    response = await client.send_message(MessageRequest(messages=messages))
    print(response.text)  # "Your name is Alice."
```

### Structured Output

```python
from pydantic import BaseModel
from smartllm import LLMClient, TextRequest

class Person(BaseModel):
    name: str
    age: int

async with LLMClient(provider="openai") as client:
    response = await client.generate_text(
        TextRequest(prompt="Return a person named John, age 30.", response_format=Person)
    )
    print(response.structured_data.name)  # "John"
```

### Streaming

```python
async with LLMClient(provider="openai") as client:
    async for chunk in client.generate_text_stream(
        TextRequest(prompt="Write a short poem.", stream=True)
    ):
        print(chunk.text, end="", flush=True)
```

### Reasoning Models

```python
response = await client.generate_text(
    TextRequest(
        prompt="Solve: what is the 100th Fibonacci number?",
        reasoning_effort="high",  # "low", "medium", or "high"
    )
)
print(response.text)
print(f"Reasoning tokens: {response.reasoning_tokens}")
```

Note: reasoning models do not support `temperature`. Passing a value other than `1` raises `ValueError`.

### OpenAI API Types

```python
# Responses API (default, recommended)
TextRequest(prompt="Hello", api_type="responses")

# Chat Completions API (legacy)
TextRequest(prompt="Hello", api_type="chat_completions")
```

### Concurrent Requests

```python
tasks = [client.generate_text(TextRequest(prompt=p)) for p in prompts]
responses = await asyncio.gather(*tasks)
```

### Progress Callbacks

```python
async def on_progress(event):
    print(event)

response = await client.generate_text(
    TextRequest(prompt="Hello", on_progress=on_progress)
)
```

Events: `llm_started`, `llm_done`, `cache_hit` (with `cache_source`, `cache_key`), `error` (with `message`). Each event dict includes `event`, `ts`, `prompt`, `model`, `provider`. `llm_done` and `cache_hit` also include `input_tokens`, `output_tokens`, `reasoning_tokens`, `cached_tokens`.

### DynamoDB Caching

```python
async with LLMClient(provider="openai", dynamo_table_name="my-llm-cache") as client:
    ...
```

Requires AWS credentials with DynamoDB access. Table is auto-created if it doesn't exist.

### Provider-Specific Clients

```python
from smartllm.openai import OpenAILLMClient, OpenAIConfig
from smartllm.bedrock import BedrockLLMClient, BedrockConfig

async with OpenAILLMClient(OpenAIConfig(api_key="...")) as client:
    models = await client.list_available_models()

async with BedrockLLMClient(BedrockConfig(aws_region="us-east-1")) as client:
    models = await client.list_available_model_ids()
```

## API Reference

### TextRequest Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| `prompt` | str | Input text prompt | Required |
| `model` | str | Model ID | Config default |
| `temperature` | float | Sampling temperature (0–1) | 0 |
| `max_tokens` | int | Maximum output tokens | 2048 |
| `top_p` | float | Nucleus sampling | 1.0 |
| `system_prompt` | str | System context | None |
| `stream` | bool | Enable streaming | False |
| `response_format` | BaseModel | Pydantic model for structured output | None |
| `use_cache` | bool | Enable caching | True |
| `clear_cache` | bool | Clear cache before request | False |
| `api_type` | str | `"responses"` or `"chat_completions"` | `"responses"` |
| `reasoning_effort` | str | `"low"`, `"medium"`, or `"high"` | None |
| `on_progress` | Callable | Progress event callback (sync or async) | None |

### TextResponse Fields

| Field | Type | Description |
|---|---|---|
| `text` | str | Generated text |
| `model` | str | Model that generated the response |
| `stop_reason` | str | Reason generation stopped |
| `input_tokens` | int | Input token count |
| `output_tokens` | int | Output token count |
| `reasoning_tokens` | int | Reasoning tokens used (OpenAI only, `0` otherwise) |
| `cached_tokens` | int | Prompt cache tokens (OpenAI only, `0` otherwise) |
| `timestamp` | str \| None | ISO 8601 UTC timestamp of the original API call |
| `elapsed_seconds` | float \| None | Duration of the original API call in seconds |
| `metadata` | dict | Request context: `prompt`/`messages` and `response_format` JSON schema |
| `structured_data` | BaseModel \| None | Parsed Pydantic object (when `response_format` was set) |
| `cache_source` | str | `"miss"`, `"l1"` (local), or `"l2"` (DynamoDB) |
| `cache_key` | str \| None | Cache key for this request |

## Caching

Responses are cached automatically when `temperature=0` or when using a reasoning model. Streaming responses are never cached.

**Cache key** is derived from: `model`, `prompt` (or `messages`), `max_tokens`, `top_p`, `system_prompt`, `response_format`, `api_type`, `reasoning_effort`.

**What is stored:**

| Field | Description |
|---|---|
| `text` | Raw response text |
| `model` | Model used |
| `stop_reason` | Stop reason |
| `input_tokens` | Input token count |
| `output_tokens` | Output token count |
| `reasoning_tokens` | Reasoning token count |
| `cached_tokens` | Prompt cache token count |
| `timestamp` | ISO 8601 UTC timestamp of the original API call |
| `elapsed_seconds` | Duration of the original API call in seconds |
| `metadata.prompt` | Original prompt (or `messages`) — stored in top-level cache metadata, not duplicated in data |
| `metadata.response_format` | JSON schema of requested output format |
| `structured_data` | Parsed Pydantic object (as dict) |

`timestamp` and `elapsed_seconds` are stored and restored on cache hits — they reflect when the original API call was made and how long it took.

```python
response1 = await client.generate_text(TextRequest(prompt="What is 2+2?", temperature=0))
print(response1.cache_source)  # "miss"

response2 = await client.generate_text(TextRequest(prompt="What is 2+2?", temperature=0))
print(response2.cache_source)  # "l1" or "l2"

# Force refresh
response3 = await client.generate_text(TextRequest(prompt="What is 2+2?", temperature=0, clear_cache=True))
```

## Development

```bash
git clone https://github.com/Redundando/smartllm.git
cd smartllm
pip install -e .[all,dev]

pytest tests/unit/ -v
pytest tests/integration/ --model gpt-4o
```

## License

MIT — see [LICENSE](LICENSE).  
Issues: [GitHub Issues](https://github.com/Redundando/smartllm/issues)
