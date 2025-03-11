# SmartLLM

SmartLLM is a unified Python interface for interacting with multiple Large Language Model providers. It provides a consistent API across different LLM providers, handles caching of responses, and supports both synchronous and streaming interactions.

## Installation

```bash
pip install smartllm
```

## Features

- **Unified API**: Consistent interface for OpenAI, Anthropic, and Perplexity LLMs
- **Response Caching**: Persistent JSON-based caching of responses to improve performance
- **Streaming Support**: Real-time streaming of LLM responses (Anthropic only)
- **JSON Mode**: Structured JSON responses (OpenAI and Anthropic)
- **Citations**: Access to source information (Perplexity only)
- **Asynchronous Execution**: Non-blocking request execution
- **Configurable Parameters**: Granular control over temperature, tokens, and other model parameters

## Supported Providers

SmartLLM currently supports the following LLM providers:

- **OpenAI**
  - Models: GPT-4, GPT-3.5 series, and other OpenAI models
  - Features: JSON-structured outputs, token usage information
  - Example: `base="openai", model="gpt-4"`

- **Anthropic**
  - Models: Claude models (e.g., claude-3-7-sonnet-20250219)
  - Features: Streaming support, JSON-structured outputs, system prompts
  - Example: `base="anthropic", model="claude-3-7-sonnet-20250219"`

- **Perplexity**
  - Models: sonar-small-online, sonar-medium-online, sonar-pro, etc.
  - Features: Web search capabilities, citation information
  - Example: `base="perplexity", model="sonar-pro"`

## Basic Usage

```python
from smartllm import SmartLLM
import os

# Create SmartLLM instance
llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY"),
    prompt="Explain quantum computing in simple terms",
    temperature=0.7
)

# Execute the request
llm.execute()

# Wait for completion
llm.wait_for_completion()

# Check status and get response
if llm.is_completed():
    print(llm.response)
else:
    print(f"Error: {llm.get_error()}")
```

## SmartLLM Class Reference

### Constructor

```python
SmartLLM(
    base: str = "",                  # LLM provider ("openai", "anthropic", "perplexity")
    model: str = "",                 # Model identifier
    api_key: str = "",               # API key for the provider
    prompt: Union[str, List[str]] = "", # Single prompt or conversation history
    stream: bool = False,            # Enable streaming (Anthropic only)
    max_input_tokens: Optional[int] = None,  # Max input tokens
    max_output_tokens: Optional[int] = None, # Max output tokens
    output_type: str = "text",       # Output type
    temperature: float = 0.2,        # Temperature for generation
    top_p: float = 0.9,              # Top-p sampling parameter
    frequency_penalty: float = 1.0,  # Frequency penalty
    presence_penalty: float = 0.0,   # Presence penalty
    system_prompt: Optional[str] = None, # System prompt
    search_recency_filter: Optional[str] = None, # Filter for search (Perplexity)
    return_citations: bool = False,  # Include citations (Perplexity)
    json_mode: bool = False,         # Enable JSON mode (OpenAI, Anthropic)
    json_schema: Optional[Dict[str, Any]] = None, # JSON schema
    ttl: int = 7,                    # Cache time-to-live in days
    clear_cache: bool = False        # Clear existing cache
)
```

### Methods

#### `execute(callback: Optional[Callable[[str, str], None]] = None) -> SmartLLM`

Initiates the LLM request. For streaming requests, an optional callback function can be provided to process each chunk of the response.

```python
# Basic execution
llm.execute()

# With streaming callback (Anthropic only)
def handle_chunk(chunk: str, accumulated: str) -> None:
    print(f"New chunk: {chunk}")

llm.execute(callback=handle_chunk)
```

#### `wait_for_completion(timeout: Optional[float] = None) -> bool`

Waits for the request to complete. Returns `True` if successful, `False` otherwise. An optional timeout parameter can be provided.

```python
# Wait indefinitely
llm.wait_for_completion()

# Wait with timeout (in seconds)
success = llm.wait_for_completion(timeout=10.0)
```

#### `is_failed() -> bool`

Returns `True` if the request failed.

#### `is_completed() -> bool`

Returns `True` if the request completed successfully.

#### `is_pending() -> bool`

Returns `True` if the request is still in progress.

#### `get_error() -> Optional[str]`

Returns the error message if the request failed, or `None` if no error occurred.

### Properties

#### `response: Union[str, Dict[str, Any]]`

Returns the generated content. If JSON mode is enabled and JSON content is available, returns a dictionary; otherwise, returns the text content.

#### `_content: str`

Returns the raw text content of the response.

#### `_json_content: Optional[Dict[str, Any]]`

Returns the parsed JSON content if available (requires `json_mode=True`).

#### `sources: List[str]`

Returns citation sources (available with Perplexity when `return_citations=True`).

#### `usage: Dict[str, int]`

Returns token usage statistics for the request, including prompt tokens, completion tokens, and total tokens.

## Advanced Features

### Streaming Responses (Anthropic Only)

```python
from smartllm import SmartLLM
import os

def print_chunk(chunk: str, accumulated: str) -> None:
    print(f"CHUNK: {chunk}")

llm = SmartLLM(
    base="anthropic",
    model="claude-3-7-sonnet-20250219",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    prompt="Write a short story about a robot learning to paint",
    stream=True  # Enable streaming
)

# Execute with callback
llm.execute(callback=print_chunk)
llm.wait_for_completion()
```

### JSON Mode (OpenAI and Anthropic)

```python
from smartllm import SmartLLM
import os

json_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 10}
    },
    "required": ["title", "topics", "difficulty"]
}

llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY"),
    prompt="Generate information about a quantum computing course",
    json_mode=True,
    json_schema=json_schema
)

llm.execute()
llm.wait_for_completion()

# Access structured data
course_info = llm.response  # Returns a Python dictionary
print(f"Course title: {course_info['title']}")
print(f"Topics: {', '.join(course_info['topics'])}")
print(f"Difficulty: {course_info['difficulty']}/10")
```

### Getting Citations (Perplexity Only)

```python
from smartllm import SmartLLM
import os

llm = SmartLLM(
    base="perplexity",
    model="sonar-pro",
    api_key=os.environ.get("PERPLEXITY_API_KEY"),
    prompt="What are the latest advancements in quantum computing?",
    search_recency_filter="week",  # Filter for recent information
    return_citations=True  # Enable citations
)

llm.execute()
llm.wait_for_completion()

# Print the response
print(llm.response)

# Print the sources
print("\nSources:")
for source in llm.sources:
    print(f"- {source}")
```

## Caching Mechanism

SmartLLM uses a persistent JSON-based caching system powered by the Cacherator library. This significantly improves performance by avoiding redundant API calls for identical requests.

### Cache Configuration

By default, responses are cached for 7 days. You can customize the cache behavior:

```python
# Set custom time-to-live (TTL) in days
llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY"),
    prompt="Explain quantum computing",
    ttl=30  # Cache results for 30 days
)

# Force clear existing cache
llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY"),
    prompt="Explain quantum computing",
    clear_cache=True  # Ignore any existing cached response
)
```

### How Caching Works

1. Each request is assigned a unique identifier based on:
   - Provider (`base`)
   - Model
   - Prompt
   - All relevant parameters (temperature, tokens, etc.)

2. Responses are stored in JSON format in the `data/llm` directory

3. When making an identical request, the cached response is returned instead of making a new API call

4. Cache entries automatically expire after the specified TTL

5. Cache can be manually cleared by setting `clear_cache=True`

## Error Handling

SmartLLM provides robust error handling through state tracking:

```python
llm = SmartLLM(...)
llm.execute()
llm.wait_for_completion()

if llm.is_failed():
    print(f"Request failed: {llm.get_error()}")
elif llm.is_completed():
    print("Request completed successfully")
    print(llm.response)
elif llm.is_pending():
    print("Request is still in progress")
```

## Dependencies

- `cacherator`: Persistent JSON-based caching
- `logorator`: Decorator-based logging
- `openai>=1.0.0`: OpenAI API client
- `anthropic>=0.5.0`: Anthropic API client
- `python-slugify`: Utility for creating safe identifiers

## License

MIT License