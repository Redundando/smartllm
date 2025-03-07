# SmartLLM

SmartLLM is a Python package that provides a unified interface for interacting with multiple Large Language Model (LLM) providers. It simplifies working with APIs from Perplexity, Anthropic, and OpenAI through a consistent interface while adding powerful features like response caching, asynchronous execution, and structured output handling.

## Features

- **Unified API**: Interact with multiple LLM providers using a consistent interface
- **Response Caching**: Avoid redundant API calls for identical requests
- **Asynchronous Execution**: Non-blocking requests with a future-based approach
- **Structured Logging**: Detailed logging of operations and requests
- **Citation Support**: Access sources/citations when available from the provider
- **JSON Mode**: Support for structured JSON output
- **Token Counting**: Provider-specific token counting or character-based estimation

## Installation

```bash
pip install smartllm
```

## Available Base LLMs

SmartLLM supports the following base LLM providers:

| Provider | Base Parameter | Description |
|----------|----------------|-------------|
| **OpenAI** | `"openai"` | GPT models from OpenAI |
| **Anthropic** | `"anthropic"` | Claude models from Anthropic |
| **Perplexity** | `"perplexity"` | Various models via Perplexity API |

Each provider requires the corresponding API key and supports different features and capabilities. Use the `list_available_models()` method to see which specific models are available from each provider.

## Basic Usage

```python
from smartllm import SmartLLM

# Create a SmartLLM instance
llm = SmartLLM(
    base="openai",  # Provider: "openai", "anthropic", or "perplexity"
    model="gpt-4o",  # Model name
    api_key="your-api-key",
    prompt="What is the tallest building in the world?",
    temperature=0.2
)

# Generate a response
llm.generate().wait_for_completion()

# Check for errors
if llm.is_failed():
    print(f"Error: {llm.get_error()}")
else:
    # Access the response content
    print(llm.content)
    
    # Access citation sources (if available)
    if llm.sources:
        for i, source in enumerate(llm.sources, 1):
            print(f"{i}. {source}")
```

## Advanced Usage

### Conversation History

You can provide a conversation history as a list of alternating messages:

```python
conversation = [
    "Hello, how can I help you today?",  # First user message
    "I'm here to assist with your questions.",  # First assistant response
    "Can you explain how solar panels work?"  # Second user message
]

llm = SmartLLM(
    base="anthropic",
    model="claude-3-7-sonnet-20250219",
    api_key="your-api-key",
    prompt=conversation
)
```

### System Prompts

Customize LLM behavior with system prompts:

```python
llm = SmartLLM(
    base="openai",
    model="gpt-4o",
    api_key="your-api-key",
    prompt="Write a poem about technology",
    system_prompt="You are a creative writing assistant with a focus on modern poetry."
)
```

### JSON Output

Request structured JSON responses:

```python
llm = SmartLLM(
    base="perplexity",
    model="sonar-pro",
    api_key="your-api-key",
    prompt="Give me information about the top 3 programming languages",
    json_mode=True,
    json_schema={
        "type": "object",
        "properties": {
            "languages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "popularity": {"type": "number"},
                        "useCases": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    }
)

llm.generate().wait_for_completion()

# Access structured JSON response
structured_data = llm.json_content
```

### Citation Retrieval

Get sources for factual information (particularly useful with Perplexity):

```python
llm = SmartLLM(
    base="perplexity",
    model="sonar-pro",
    api_key="your-api-key",
    prompt="What are the recent advancements in quantum computing?",
    return_citations=True
)

llm.generate().wait_for_completion()

# Access content and sources
print(llm.content)
for source in llm.sources:
    print(f"Source: {source}")
```

### Token Usage Information

Access token usage metrics:

```python
llm.generate().wait_for_completion()
usage = llm.usage

print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
```

### Request State Management

Check the status of your request:

```python
llm.generate()  # Start generating asynchronously

# Check state
if llm.is_pending():
    print("Request is still processing...")
elif llm.is_completed():
    print("Request completed successfully")
elif llm.is_failed():
    print(f"Request failed: {llm.get_error()}")
```

## API Reference

### SmartLLM Class

The main class for interacting with LLM providers.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base` | `str` | `""` | Provider name: "perplexity", "anthropic", or "openai" |
| `model` | `str` | `""` | Model name for the selected provider |
| `api_key` | `str` | `""` | API key for the selected provider |
| `prompt` | `Union[str, List[str]]` | `""` | Text prompt or conversation history |
| `max_input_tokens` | `Optional[int]` | `10000` | Maximum input tokens |
| `max_output_tokens` | `Optional[int]` | `10000` | Maximum output tokens |
| `temperature` | `float` | `0.2` | Temperature for response generation (0.0-1.0) |
| `top_p` | `float` | `0.9` | Top-p sampling value |
| `frequency_penalty` | `float` | `1.0` | Frequency penalty value |
| `presence_penalty` | `float` | `0.0` | Presence penalty value |
| `system_prompt` | `Optional[str]` | `None` | System prompt for setting context or behavior |
| `search_recency_filter` | `Optional[str]` | `None` | Filter for search recency ("month", "week", "day", "hour") |
| `return_citations` | `bool` | `False` | Whether to return citation sources |
| `json_mode` | `bool` | `False` | Whether to request JSON output |
| `json_schema` | `Optional[Dict[str, Any]]` | `None` | Schema for structured JSON output |

#### Methods

| Method | Description |
|--------|-------------|
| `generate()` | Starts the generation process asynchronously |
| `wait_for_completion(timeout=None)` | Waits for the request to complete with optional timeout |
| `is_completed()` | Checks if the request has completed successfully |
| `is_pending()` | Checks if the request is still in progress |
| `is_failed()` | Checks if the request has failed |
| `get_error()` | Returns the error message if the request failed |
| `count_tokens()` | Estimates the token count for the current prompt |
| `list_available_models(limit=20)` | Lists available models for the current provider |

#### Properties

| Property | Description |
|----------|-------------|
| `content` | The text content of the response |
| `json_content` | The structured JSON content (when `json_mode=True`) |
| `sources` | List of citation sources (when available) |
| `usage` | Token usage statistics |

### Static Utilities

| Method | Description |
|--------|-------------|
| `convert_schema(schema, provider=None)` | Converts a schema to a provider-specific format |

## Logging and Caching

### Logging

SmartLLM uses the Logorator library for comprehensive logging of operations:

- **Hierarchical Logging**: Nested function calls are properly indented in logs
- **Function Call Tracking**: Automatically logs when functions are called with their parameters
- **Execution Time Measurement**: Tracks how long operations take to complete
- **Note Adding**: Add custom notes to your logs using `Logger.note()` from the Logorator package

### Caching

SmartLLM leverages the Cacherator library for efficient response caching:

- **Persistent Caching**: Responses are cached to disk and reused between program executions
- **Automatic Cache Management**: Identical requests automatically use cached responses
- **Time-To-Live (TTL)**: Cache entries can expire after a specified time
- **JSON-based Storage**: Cache is stored in an easily inspectable JSON format

This caching mechanism significantly reduces API costs and improves response times for repeated queries.

## Error Handling

Always check for request failures before accessing response properties:

```python
llm.generate().wait_for_completion()

if llm.is_failed():
    print(f"Request failed: {llm.get_error()}")
    # Handle error appropriately
else:
    # Process successful response
    print(llm.content)
```

## Dependencies

- **cacherator**: For persistent JSON-based caching
- **logorator**: For structured logging
- **Provider-specific client libraries**: openai, anthropic

## License

MIT License