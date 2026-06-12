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
- **Streaming with Assembly** — Internal streaming that returns a single `TextResponse` (solves Bedrock read timeouts on large requests)
- **Rate Limiting** — Built-in concurrency control per model
- **Reasoning Models** — Full support including `reasoning_effort` and `reasoning_tokens`
- **Extended Thinking (Bedrock)** — Claude extended thinking with two-pass structured output. Auto-handles both manual-budget (Sonnet 3.7–4.6, Opus 4.5) and adaptive-effort (Opus 4.6+) APIs.
- **Bedrock Model Capability Awareness** — Per-model body construction. The package detects what each Claude model accepts (sampling params, thinking shape) and adapts the request automatically. Same calling code works across Claude 3.x through Opus 4.7+.
- **Progress Callbacks** — Optional `on_progress` for real-time events (including retries)
- **Configurable Timeouts** — Adjustable HTTP read/connect timeouts for Bedrock (default 300s read)

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
export AWS_REGION="eu-north-1"
export BEDROCK_MODEL="eu.anthropic.claude-sonnet-4-6"  # optional (use an inference profile ID)
export BEDROCK_READ_TIMEOUT="300"   # HTTP read timeout in seconds (default: 300)
export BEDROCK_CONNECT_TIMEOUT="10" # HTTP connect timeout in seconds (default: 10)
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

### Streaming with Assembly (Bedrock only)

`generate_text_streamed` uses Bedrock's streaming API internally but returns a fully assembled `TextResponse` — identical to `generate_text()`. This solves read timeouts on large requests (50K+ input, 16K+ output tokens) where the non-streaming `invoke_model` connection idles and times out.

```python
async with LLMClient(provider="bedrock") as client:
    response = await client.generate_text_streamed(
        TextRequest(
            prompt="Write a 5000-word technical analysis...",
            max_tokens=8192,
            temperature=0,
        )
    )
    # Returns a normal TextResponse — no chunk iteration needed
    print(response.text)
    print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
```

**When to use `generate_text_streamed` vs `generate_text`:**

| Scenario | Method |
|----------|--------|
| Short requests (< 30K chars input, < 4K tokens output) | `generate_text` |
| Large requests that risk read timeout (long generation time) | `generate_text_streamed` |
| Need structured output (`response_format`) | `generate_text` (streamed rejects this) |
| Need progress visibility during long generation | `generate_text_streamed` |
| OpenAI provider | `generate_text` (streamed is Bedrock-only) |

**Behavior:**
- Same `TextResponse` shape as `generate_text` (text, model, tokens, metadata, cache)
- Same cache keys — a response cached by one method is served to the other
- Same semaphore, retry logic, and concurrency gating
- Fires progress events: `llm_started`, `stream_progress`, `stream_thinking`, `llm_done`, `error`, `retry`, `cache_hit`
- Raises `ValueError` if `response_format` is set (suggests `generate_text` as alternative)
- Raises `NotImplementedError` on OpenAI provider

**Progress events during streaming:**

```python
def on_progress(event):
    if event["event"] == "stream_progress":
        print(f"{event['text_tokens_so_far']} tokens generated...")
    elif event["event"] == "stream_thinking":
        print(f"{event['thinking_tokens_so_far']} thinking tokens...")

response = await client.generate_text_streamed(
    TextRequest(prompt="...", on_progress=on_progress)
)
```

`stream_progress` and `stream_thinking` fire every ~500 estimated tokens or every 10 seconds (whichever comes first). Token count is estimated as `len(text) // 4`.

| Event | Fields |
|-------|--------|
| `stream_progress` | `text_tokens_so_far`, `text_so_far`, `elapsed_seconds` |
| `stream_thinking` | `thinking_tokens_so_far`, `thinking_text_so_far`, `elapsed_seconds` |

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

### Extended Thinking (Bedrock/Claude)

Claude models on Bedrock support extended thinking, where the model reasons step-by-step before answering. The package handles two different thinking APIs transparently — pick the model you want and the request is shaped correctly.

**How it works under the hood:**

| Claude generation | Sampling params (`temperature`, `top_p`, `top_k`) | Thinking shape | Notes |
|---|---|---|---|
| Sonnet 3.x, Opus 3.x | accepted | not supported (silently ignored) | sampling unchanged |
| Sonnet 3.7 | accepted | manual budget (`thinking.type=enabled`, `budget_tokens=N`) | classic shape |
| Sonnet 4.x, Opus 4.5 | accepted | manual budget | classic shape |
| Sonnet 4.6 | accepted | manual budget | classic shape |
| **Opus 4.6** | accepted | **adaptive** (`thinking.type=adaptive`, `output_config.effort=...`) | model decides depth |
| **Opus 4.7, 4.8** | **rejected** (dropped with a warning) | **adaptive** | sampling controls deprecated |

You don't need to know which generation supports which shape — pass `reasoning_effort` (or `budget_tokens`) and the package emits the right body. Sampling parameters that the target model rejects are dropped with a `Logger.warning` so the call doesn't fail.

**Common usage:**

```python
async with LLMClient(provider="bedrock") as client:
    # Works identically across Claude generations.
    response = await client.generate_text(
        TextRequest(
            prompt="Analyze the tradeoffs of event sourcing vs CRUD.",
            model="eu.anthropic.claude-sonnet-4-6",   # or eu.anthropic.claude-opus-4-7
            reasoning_effort="high",                  # "low" | "medium" | "high"
        )
    )
    print(response.text)
    print(f"Reasoning tokens: {response.reasoning_tokens}")
    print(f"Thinking trace: {response.metadata.get('thinking', '')[:200]}")
```

For precise control on **manual-budget** models, use `budget_tokens` directly (overrides `reasoning_effort` mapping):

```python
response = await client.generate_text(
    TextRequest(
        prompt="Solve this step by step...",
        model="eu.anthropic.claude-sonnet-4-6",
        budget_tokens=8192,  # minimum 1024
    )
)
```

On **adaptive** models (Opus 4.6+) `budget_tokens` has no direct equivalent — it's mapped to the nearest effort level (`low`/`medium`/`high`) with a warning. Prefer `reasoning_effort` for those.

**`reasoning_effort` to budget mapping** (manual-budget models only):

| Effort | `budget_tokens` |
|---|---|
| `low` | 1024 |
| `medium` | 4096 |
| `high` | 16000 |

#### Capability Inspection

Inspect what a model accepts without making a call:

```python
from smartllm.bedrock import BedrockLLMClient
from smartllm.bedrock.capabilities import get_model_capabilities, supports_thinking

caps = get_model_capabilities("eu.anthropic.claude-opus-4-7")
# ModelCapabilities(
#     family='claude-opus-4-7',
#     accepts_temperature=False,
#     accepts_top_p_top_k=False,
#     thinking_mode='adaptive_effort',
# )

supports_thinking("us.anthropic.claude-3-5-sonnet-20241022-v2:0")  # False
supports_thinking("eu.anthropic.claude-sonnet-4-6")                # True

# Equivalent staticmethods on the client:
BedrockLLMClient.get_model_capabilities("...")
BedrockLLMClient.supports_thinking("...")
```

`thinking_mode` is one of `"none"`, `"manual_budget"`, or `"adaptive_effort"`. Use this to decide upfront whether to set `reasoning_effort` on a request.

#### Extended Thinking + Structured Output

When both `reasoning_effort` (or `budget_tokens`) and `response_format` are set, SmartLLM uses a two-pass approach:

1. **Pass 1** — Sends the prompt with extended thinking enabled. Claude reasons through the problem and produces a text answer.
2. **Pass 2** — Sends the text answer to a second call with forced tool use to extract it into the Pydantic model. The pass-2 prompt instructs the model to return native JSON arrays/objects (mitigates a Bedrock quirk on non-English content).

```python
from pydantic import BaseModel
from typing import List

class Analysis(BaseModel):
    topic: str
    pros: List[str]
    cons: List[str]
    recommendation: str

response = await client.generate_text(
    TextRequest(
        prompt="Should we use microservices or a monolith?",
        model="eu.anthropic.claude-sonnet-4-6",
        reasoning_effort="medium",
        response_format=Analysis,
    )
)
print(response.structured_data.recommendation)
print(response.metadata["pass1_tokens"])  # {"input": ..., "output": ...}
print(response.metadata["pass2_tokens"])  # {"input": ..., "output": ...}
```

The two-pass approach is needed because Claude's extended thinking is incompatible with forced tool use (`tool_choice: {"type": "tool"}`). The result is cached as a single entry — on cache hit, both passes are skipped.

#### Streaming with Extended Thinking

When streaming with thinking enabled, thinking chunks are yielded with `metadata={"type": "thinking"}`:

```python
async for chunk in client.generate_text_stream(
    TextRequest(prompt="Explain quantum entanglement.", reasoning_effort="medium", stream=True)
):
    if chunk.metadata.get("type") == "thinking":
        print(f"[thinking] {chunk.text}", end="")
    else:
        print(chunk.text, end="")
```

#### Multi-turn Conversations with Thinking

`MessageRequest` supports the same thinking parameters as `TextRequest`:

```python
from smartllm import LLMClient, MessageRequest, Message

async with LLMClient(provider="bedrock") as client:
    messages = [
        Message(role="user", content="I'm planning a 2-week trip to Japan."),
        Message(role="assistant", content="Great! What's your budget and what interests you?"),
        Message(role="user", content="$3000, history and food. Plan a rough itinerary."),
    ]
    response = await client.send_message(
        MessageRequest(
            messages=messages,
            model="eu.anthropic.claude-opus-4-7",
            reasoning_effort="medium",
        )
    )
    print(response.text)
```

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

Events: `llm_started`, `llm_done`, `cache_hit` (with `cache_source`, `cache_key`), `retry`, `error` (with `message`). Each event dict includes `event`, `ts`, `prompt`, `model`, `provider`. `llm_done` and `cache_hit` also include `input_tokens`, `output_tokens`, `reasoning_tokens`, `cached_tokens`.

The `retry` event is emitted before each retry attempt and includes:

| Field | Description |
|---|---|
| `event` | `"retry"` |
| `attempt` | Current retry number (1-indexed) |
| `max_retries` | Total retries configured |
| `error` | Exception class name (e.g. `"ReadTimeoutError"`) |
| `error_message` | Full error string |
| `model` | Model being called |
| `max_tokens` | Max tokens for this request |
| `delay` | Seconds until next attempt |

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

async with BedrockLLMClient(BedrockConfig(aws_region="us-east-1", read_timeout=300)) as client:
    models = await client.list_available_model_ids()
```

## API Reference

### TextRequest Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| `prompt` | str | Input text prompt | Required |
| `model` | str | Model ID (or Bedrock inference profile ID) | Config default |
| `temperature` | float | Sampling temperature (0–1). Auto-dropped on Opus 4.7+. | 0 |
| `max_tokens` | int | Maximum output tokens | 2048 |
| `top_p` | float | Nucleus sampling. Forwarded to Claude when supported (auto-dropped on Opus 4.7+). | None (model default) |
| `top_k` | int | Top-k sampling (Bedrock only). Forwarded to Claude when supported (auto-dropped on Opus 4.7+). | None |
| `system_prompt` | str | System context | None |
| `stream` | bool | Enable streaming | False |
| `response_format` | BaseModel | Pydantic model for structured output | None |
| `use_cache` | bool | Enable caching | True |
| `clear_cache` | bool | Clear cache before request | False |
| `api_type` | str | `"responses"` or `"chat_completions"` | `"responses"` |
| `reasoning_effort` | str | `"low"`, `"medium"`, or `"high"` | None |
| `budget_tokens` | int | Explicit thinking budget for Bedrock manual-budget models. Mapped to nearest effort on adaptive (Opus 4.6+) models. Minimum 1024. | None |
| `on_progress` | Callable | Progress event callback (sync or async) | None |

### MessageRequest Parameters

`MessageRequest` is used for multi-turn conversations via `send_message` / `send_message_stream`. It mirrors `TextRequest` but takes a `messages` list instead of a `prompt`.

| Parameter | Type | Description | Default |
|---|---|---|---|
| `messages` | list[Message] | Conversation history (`role` is `"user"` or `"assistant"`) | Required |
| `model` | str | Model ID | Config default |
| `temperature` | float | Sampling temperature. Auto-dropped on Opus 4.7+. | 0 |
| `max_tokens` | int | Maximum output tokens | 2048 |
| `top_p` | float | Nucleus sampling. Forwarded to Claude when supported. | None |
| `top_k` | int | Top-k sampling (Bedrock only). Forwarded to Claude when supported. | None |
| `system_prompt` | str | System context | None |
| `stream` | bool | Enable streaming | False |
| `response_format` | BaseModel | Pydantic model for structured output | None |
| `use_cache` | bool | Enable caching | True |
| `clear_cache` | bool | Clear cache before request | False |
| `api_type` | str | `"responses"` or `"chat_completions"` | `"responses"` |
| `reasoning_effort` | str | `"low"`, `"medium"`, or `"high"` (Bedrock Claude with thinking support) | None |
| `budget_tokens` | int | Explicit thinking budget. Same semantics as on `TextRequest`. | None |
| `on_progress` | Callable | Progress event callback (sync or async) | None |

### TextResponse Fields

| Field | Type | Description |
|---|---|---|
| `text` | str | Generated text |
| `model` | str | Model that generated the response |
| `stop_reason` | str | Reason generation stopped |
| `input_tokens` | int | Input token count |
| `output_tokens` | int | Output token count |
| `reasoning_tokens` | int | Reasoning/thinking tokens used (OpenAI reasoning models and Bedrock extended thinking) |
| `cached_tokens` | int | Prompt cache tokens (OpenAI only, `0` otherwise) |
| `timestamp` | str \| None | ISO 8601 UTC timestamp of the original API call |
| `elapsed_seconds` | float \| None | Duration of the original API call in seconds |
| `metadata` | dict | Request context: `prompt`/`messages` and `response_format` JSON schema |
| `structured_data` | BaseModel \| None | Parsed Pydantic object (when `response_format` was set) |
| `cache_source` | str | `"miss"`, `"l1"` (local), or `"l2"` (DynamoDB) |
| `cache_key` | str \| None | Cache key for this request |

## Structured Output Error Handling

When using `response_format`, two error conditions are raised explicitly:

**Truncated output** — if the provider cuts off the response before the structured output is complete, a `ValueError` is raised:

```python
try:
    response = await client.generate_text(
        TextRequest(prompt="...", response_format=MyModel, max_tokens=100)
    )
except ValueError as e:
    print(e)  # "Bedrock truncated structured output (stop_reason=max_tokens)"
             # "OpenAI truncated structured output (finish_reason=length)"
             # "OpenAI truncated structured output (status=incomplete)"
```

Increase `max_tokens` to avoid this.

**Provider serialization quirks** — Bedrock occasionally returns list/dict fields inside a tool-use payload as JSON-encoded strings rather than native arrays/objects. Most often observed on Sonnet 4.6 with non-English content (e.g. German). SmartLLM handles this automatically: `_parse_response` first attempts strict Pydantic validation, then retries after `json.loads`-ing any string fields that look like a JSON array or object. The two-pass thinking + structure flow also instructs the model to emit native arrays/objects.

If your model has list fields and you still see `ValidationError` after the tolerant retry (e.g. nested fragmentation, deeply malformed payloads), add a field validator:

```python
import json
from pydantic import BaseModel, field_validator

class BookList(BaseModel):
    books: list[str]

    @field_validator("books", mode="before")
    @classmethod
    def parse_json_string(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
```

## Caching

Responses are cached automatically when `temperature=0`, when using a reasoning model, or when extended thinking is enabled. Streaming responses (`generate_text_stream`) are never cached. `generate_text_streamed` responses are cached — they share the same cache keys as `generate_text`.

**Cache key** is derived from: `model`, `prompt` (or `messages`), `max_tokens`, `top_p`, `system_prompt`, `response_format`, `api_type`, `reasoning_effort`, `budget_tokens`.

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
