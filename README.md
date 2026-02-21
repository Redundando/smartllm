# SmartLLM

A unified async Python wrapper for multiple LLM providers with a consistent interface.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Unified Interface** - Single API for multiple LLM providers (OpenAI, AWS Bedrock)
- **Async/Await** - Built on asyncio for high-performance concurrent requests
- **Smart Caching** - Two-level cache (local + DynamoDB) to reduce costs and latency
- **Auto Retry** - Exponential backoff retry logic for transient failures
- **Structured Output** - Native Pydantic model support for type-safe responses
- **Streaming** - Real-time streaming responses for better UX
- **Rate Limiting** - Built-in concurrency control per model
- **Decorator Logging** - Automatic function logging via [Logorator](https://pypi.org/project/logorator/)
- **OpenAI Response API** - Full support for OpenAI's primary API including reasoning models

## Installation

```bash
pip install smartllm
```

### Optional Dependencies

Install only the providers you need:

```bash
# For OpenAI
pip install smartllm[openai]

# For AWS Bedrock
pip install smartllm[bedrock]

# For all providers
pip install smartllm[all]
```

### DynamoDB Caching (optional)

To enable shared two-level caching across machines:

```python
async with LLMClient(provider="openai", dynamo_table_name="my-llm-cache") as client:
    ...
```

Requires AWS credentials with DynamoDB access. The table is auto-created if it doesn't exist. Local file cache is always used as the first layer.

## Quick Start

### Basic Usage

```python
import asyncio
from smartllm import LLMClient, TextRequest

async def main():
    # Auto-detects provider from environment variables
    async with LLMClient(provider="openai") as client:
        response = await client.generate_text(
            TextRequest(prompt="What is the capital of France?")
        )
        print(response.text)

asyncio.run(main())
```

### Multi-turn Conversations

```python
from smartllm import LLMClient, MessageRequest, Message

async with LLMClient(provider="openai") as client:
    messages = [
        Message(role="user", content="My name is Alice."),
        Message(role="assistant", content="Nice to meet you, Alice!"),
        Message(role="user", content="What's my name?"),
    ]
    
    response = await client.send_message(
        MessageRequest(messages=messages)
    )
    print(response.text)  # "Your name is Alice."
```

### Streaming Responses

```python
from smartllm import LLMClient, TextRequest

async with LLMClient(provider="openai") as client:
    request = TextRequest(
        prompt="Write a short poem about Python.",
        stream=True
    )
    
    async for chunk in client.generate_text_stream(request):
        print(chunk.text, end="", flush=True)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from smartllm import LLMClient, TextRequest

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async with LLMClient(provider="openai") as client:
    response = await client.generate_text(
        TextRequest(
            prompt="Generate a person profile for a software engineer named John, age 30.",
            response_format=Person
        )
    )
    
    person = response.structured_data
    print(f"{person.name} is a {person.age} year old {person.occupation}")
```

## Configuration

### Environment Variables

**OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # Optional
```

**AWS Bedrock:**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
export BEDROCK_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"  # Optional
```

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
    # Use client...
    pass
```

### Customizing Defaults

```python
from smartllm import defaults

# Modify global defaults
defaults.DEFAULT_TEMPERATURE = 0.7
defaults.DEFAULT_MAX_TOKENS = 4096
defaults.DEFAULT_MAX_RETRIES = 5
```

### OpenAI API Types

SmartLLM supports both OpenAI APIs via the `api_type` parameter:

- `"responses"` (default) - OpenAI's primary [Response API](https://platform.openai.com/docs/api-reference/responses), recommended for all modern models
- `"chat_completions"` - Legacy [Chat Completions API](https://platform.openai.com/docs/api-reference/chat), supported indefinitely

```python
# Response API (default)
response = await client.generate_text(
    TextRequest(prompt="Hello", api_type="responses")
)

# Chat Completions API (legacy)
response = await client.generate_text(
    TextRequest(prompt="Hello", api_type="chat_completions")
)
```

### Reasoning Models

For models that support reasoning (e.g. GPT-5.x), use `reasoning_effort` to control how much the model reasons before responding. Reasoning tokens are returned in `response.metadata`:

```python
response = await client.generate_text(
    TextRequest(
        prompt="Solve: what is the 100th Fibonacci number?",
        reasoning_effort="high",  # "low", "medium", or "high"
    )
)

print(response.text)
print(f"Reasoning tokens used: {response.metadata.get('reasoning_tokens', 0)}")
```

Note: reasoning models do not support `temperature`. Passing a value other than `1` will raise a `ValueError`.

### Reasoning with Structured Output

```python
from pydantic import BaseModel
from smartllm import LLMClient, TextRequest

class Solution(BaseModel):
    answer: float
    unit: str
    explanation: str

async with LLMClient(provider="openai") as client:
    response = await client.generate_text(
        TextRequest(
            prompt="A train leaves city A at 60mph toward city B (300 miles away). Another leaves B at 90mph. When do they meet?",
            response_format=Solution,
            reasoning_effort="medium",
        )
    )

    solution = response.structured_data
    print(f"{solution.answer} {solution.unit}: {solution.explanation}")
    print(f"Reasoning tokens: {response.metadata.get('reasoning_tokens', 0)}")
```

## Advanced Features

### Caching

Responses are automatically cached when `temperature=0`:

```python
# First call - hits API
response1 = await client.generate_text(
    TextRequest(prompt="What is 2+2?", temperature=0)
)

# Second call - uses cache (instant, free)
response2 = await client.generate_text(
    TextRequest(prompt="What is 2+2?", temperature=0)
)

# Clear cache for specific request
response3 = await client.generate_text(
    TextRequest(prompt="What is 2+2?", temperature=0, clear_cache=True)
)
```

### Concurrent Requests

```python
import asyncio
from smartllm import LLMClient, TextRequest

async with LLMClient(provider="openai") as client:
    prompts = ["Question 1", "Question 2", "Question 3"]
    
    tasks = [
        client.generate_text(TextRequest(prompt=p))
        for p in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
```

### Rate Limiting

```python
# Limit concurrent requests
client = LLMClient(provider="openai", max_concurrent=5)
```

### Provider-Specific Clients

For advanced use cases, access provider-specific clients:

```python
from smartllm.openai import OpenAILLMClient, OpenAIConfig
from smartllm.bedrock import BedrockLLMClient, BedrockConfig

# OpenAI-specific features
openai_config = OpenAIConfig(api_key="...", organization="...")
async with OpenAILLMClient(openai_config) as client:
    models = await client.list_available_models()

# Bedrock-specific features
bedrock_config = BedrockConfig(aws_region="us-east-1")
async with BedrockLLMClient(bedrock_config) as client:
    models = await client.list_available_model_ids()
```

## Supported Providers

- **OpenAI** - GPT models via OpenAI API
- **AWS Bedrock** - Claude, Llama, Mistral, and Titan models

## API Reference

### Core Classes

- **`LLMClient`** - Unified client for all providers
- **`LLMConfig`** - Unified configuration
- **`TextRequest`** - Single prompt request
- **`MessageRequest`** - Multi-turn conversation request
- **`TextResponse`** - LLM response with metadata
- **`Message`** - Conversation message
- **`StreamChunk`** - Streaming response chunk

### Request Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | str | Input text prompt | Required |
| `model` | str | Model ID to use | Config default |
| `temperature` | float | Sampling temperature (0-1) | 0 |
| `max_tokens` | int | Maximum output tokens | 2048 |
| `top_p` | float | Nucleus sampling | 1.0 |
| `system_prompt` | str | System context | None |
| `stream` | bool | Enable streaming | False |
| `response_format` | BaseModel | Pydantic model for structured output | None |
| `use_cache` | bool | Enable caching | True |
| `clear_cache` | bool | Clear cache before request | False |
| `api_type` | str | OpenAI API type (`"responses"` or `"chat_completions"`) | `"responses"` |
| `reasoning_effort` | str | Reasoning effort (`"low"`, `"medium"`, `"high"`) | None |

## Error Handling

```python
from smartllm import LLMClient, TextRequest

async with LLMClient(provider="openai") as client:
    try:
        response = await client.generate_text(
            TextRequest(prompt="Hello")
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"API error: {e}")
```

## Development

### Setup

```bash
git clone https://github.com/Redundando/smartllm.git
cd smartllm
pip install -e .[all,dev]
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (select model interactively)
pytest tests/integration/

# Integration tests with a specific model
pytest tests/integration/ --model gpt-4o

# Integration tests with a reasoning model
pytest tests/integration/ --model gpt-5.2
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 0.1.5
- Replaced custom logging with [Logorator](https://pypi.org/project/logorator/) decorator-based logging
- Added two-level cache: local JSON files + optional DynamoDB via [Dynamorator](https://pypi.org/project/dynamorator/)
- DynamoDB cache configurable via `dynamo_table_name` and `cache_ttl_days` (default: 365 days)
- Cache write-back: DynamoDB hits are written to local cache automatically
- Prompt stored in cache metadata
- Recursive Pydantic schema cleaning for OpenAI structured output compatibility
- `logorator` and `dynamorator` added as core dependencies in `pyproject.toml`

### Version 0.1.4
- Fixed logger name from `aws_llm_wrapper` to `smartllm`
- Removed redundant `response_format=json_object` when using tool-based structured output
- Cache read failures now log a warning instead of silently returning `None`
- Added `reasoning_effort` warning when used with Bedrock models
- Test suite now supports model selection via `--model` CLI option or interactive prompt
- Integration tests support both OpenAI and AWS Bedrock models
- Bedrock streaming chunk parsing fixed for Claude models

### Version 0.1.0
- Initial public release
- Unified interface for multiple providers
- OpenAI support (GPT models)
- AWS Bedrock support (Claude, Llama, Mistral, Titan)
- Async/await architecture
- Smart caching with temperature=0
- Auto retry with exponential backoff
- Structured output with Pydantic models
- Streaming responses
- Rate limiting and concurrency control
- OpenAI Response API support (primary interface)
- Reasoning model support with `reasoning_effort` parameter
- Comprehensive test suite

## Support

- **Issues**: [GitHub Issues](https://github.com/Redundando/smartllm/issues)
- **Email**: arved.kloehn@gmail.com

## Acknowledgments

Built with:
- [Pydantic](https://pydantic.dev/) for data validation
- [Logorator](https://pypi.org/project/logorator/) for decorator-based logging
- [Dynamorator](https://pypi.org/project/dynamorator/) for DynamoDB caching
- [aioboto3](https://github.com/terrycain/aioboto3) for AWS async support
- [OpenAI Python SDK](https://github.com/openai/openai-python) for OpenAI integration
