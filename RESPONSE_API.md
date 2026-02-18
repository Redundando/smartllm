# OpenAI Response API Support

## Overview

SmartLLM now supports OpenAI's new **Response API** (the primary interface for GPT-5+ models) alongside the legacy Chat Completions API.

## API Types

### Response API (Default)
- **Primary interface** for modern OpenAI models (GPT-5.x)
- Simplified parameters: `input`, `instructions`, `max_output_tokens`
- Token usage: `input_tokens`, `output_tokens`
- Structured output via `text.format` with JSON schema

### Chat Completions API (Legacy)
- Supported indefinitely by OpenAI
- Traditional `messages` array format
- Token usage: `prompt_tokens`, `completion_tokens`
- Structured output via `response_format` and `tools`

## Usage

### Using Response API (Default)

```python
from smartllm import LLMClient, TextRequest

async with LLMClient(provider="openai") as client:
    response = await client.generate_text(
        TextRequest(
            prompt="Write a haiku about AI.",
            model="gpt-5.2",
            api_type="responses"  # Default, can be omitted
        )
    )
    print(response.text)
```

### Using Chat Completions API

```python
from smartllm import LLMClient, TextRequest

async with LLMClient(provider="openai") as client:
    response = await client.generate_text(
        TextRequest(
            prompt="Write a haiku about AI.",
            model="gpt-4o",
            api_type="chat_completions"  # Explicit
        )
    )
    print(response.text)
```

## Parameter Mapping

### TextRequest → Response API
- `prompt` → `input`
- `system_prompt` → `instructions`
- `max_tokens` → `max_output_tokens`
- `temperature` → `temperature`
- `top_p` → `top_p`
- `response_format` → `text.format` (JSON schema)

### TextRequest → Chat Completions API
- `prompt` → `messages[{role: "user", content: ...}]`
- `system_prompt` → `messages[{role: "system", content: ...}]`
- `max_tokens` → `max_tokens`
- `temperature` → `temperature`
- `top_p` → `top_p`
- `response_format` → `response_format` + `tools`

## Token Usage

### Response API
```python
response.input_tokens   # Input tokens consumed
response.output_tokens  # Output tokens generated
```

### Chat Completions API
```python
response.input_tokens   # Mapped from prompt_tokens
response.output_tokens  # Mapped from completion_tokens
```

## Streaming

Streaming support for Response API will be added in a future update. Currently, streaming only works with Chat Completions API.

## Model Compatibility

- **GPT-5.x models**: Use Response API (default)
- **GPT-4.x models**: Use either API (Chat Completions recommended for compatibility)
- **GPT-3.5 models**: Use Chat Completions API

## Implementation Status

✅ Response API - Text generation  
✅ Response API - Structured output  
✅ Response API - Token counting  
✅ Chat Completions API - Full support  
⏳ Response API - Streaming (coming soon)  
⏳ Response API - Multi-turn conversations (coming soon)
