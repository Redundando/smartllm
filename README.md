# SmartLLM

SmartLLM is a unified Python interface for interacting with multiple Large Language Model providers. It provides a consistent API across different LLM providers, handles caching of responses, and supports both synchronous and streaming interactions.

## Installation

```bash
pip install smartllm
```

## Features

- **Unified API**: Consistent interface for OpenAI, Anthropic, and Perplexity LLMs
- **Response Caching**: Persistent JSON-based caching of responses to improve performance
- **Streaming Support**: Real-time streaming of LLM responses with callback functionality
- **Asynchronous Execution**: Non-blocking request execution
- **Configurable Parameters**: Granular control over temperature, tokens, and other model parameters
- **JSON Mode Support**: Built-in handling for JSON-structured responses
- **Robust Error Handling**: Comprehensive error management and state tracking

## Core Components

### SmartLLM

The primary class for interacting with LLM providers:

```python
from smartllm import SmartLLM

llm = SmartLLM(
        base="openai",  # Provider: "openai", "anthropic", or "perplexity"
        model="gpt-4",  # Model name
        api_key="your-api-key",
        prompt="Your prompt here",
        temperature=0.7
)

llm.generate()  # Start generation process
llm.wait_for_completion()  # Wait for the response
result = llm._content  # Get the generated content
```

### StreamingLLM

For real-time streaming of responses:

```python
from smartllm import SmartLLM

llm = SmartLLM(
    base="anthropic",
    model="claude-3-7-sonnet-20250219",
    api_key="your-api-key",
    prompt="Your prompt here",
    stream=True
)

# Stream with custom callback function
def handle_chunk(chunk, accumulated):
    print(f"New chunk: {chunk}")

llm.stream(callback=handle_chunk)
llm.wait_for_completion()
```

## API Reference

### SmartLLM Class

#### Constructor

```python
SmartLLM(
    base: str = "",                  # LLM provider ("openai", "anthropic", "perplexity")
    model: str = "",                 # Model identifier
    api_key: str = "",               # API key for the provider
    prompt: Union[str, List[str]] = "", # Single prompt or conversation history
    stream: bool = False,            # Enable streaming support
    max_input_tokens: Optional[int] = None,  # Max input tokens
    max_output_tokens: Optional[int] = None, # Max output tokens
    output_type: str = "text",       # Output type
    temperature: float = 0.2,        # Temperature for generation
    top_p: float = 0.9,              # Top-p sampling parameter
    frequency_penalty: float = 1.0,  # Frequency penalty
    presence_penalty: float = 0.0,   # Presence penalty
    system_prompt: Optional[str] = None, # System prompt
    search_recency_filter: Optional[str] = None, # Filter for search providers
    return_citations: bool = False,  # Include citations in response
    json_mode: bool = False,         # Enable JSON mode
    json_schema: Optional[Dict[str, Any]] = None, # JSON schema
    ttl: int = 7,                    # Cache time-to-live in days
    clear_cache: bool = False        # Clear existing cache
)
```

#### Methods

##### `generate() -> SmartLLM`

Initiates the LLM request. Returns the SmartLLM instance for chaining.

##### `stream(callback: Optional[Callable[[str, str], None]] = None) -> SmartLLM`

Initiates a streaming request. The callback receives each chunk and the accumulated content.

##### `wait_for_completion(timeout: Optional[float] = None) -> bool`

Waits for the request to complete. Returns True if successful, False otherwise.

##### `is_failed() -> bool`

Returns True if the request failed.

##### `is_completed() -> bool`

Returns True if the request completed successfully.

##### `is_pending() -> bool`

Returns True if the request is still in progress.

##### `get_error() -> Optional[str]`

Returns the error message if the request failed.

##### `_get_llm_response() -> Dict[str, Any]` (Cached)

Internal method that handles the actual LLM request. Results are cached based on the configuration parameters.

##### `_get_streaming_llm_response() -> Dict[str, Any]` (Cached)

Internal method that handles streaming LLM requests. Results are cached based on the configuration parameters.

#### Properties

##### `content: str` (Uses cached data)

The generated content from the LLM. Retrieved from cached response.

##### `json_content: Optional[Dict[str, Any]]` (Uses cached data)

The parsed JSON content if json_mode is enabled. Retrieved from cached response.

##### `sources: List[str]` (Uses cached data)

Citations or sources included in the response. Retrieved from cached response.

##### `usage: Dict[str, int]` (Uses cached data)

Token usage statistics for the request. Retrieved from cached response.

### Configuration Options

#### Available Providers

- `"openai"`: OpenAI models (e.g., GPT-4, GPT-3.5)
- `"anthropic"`: Anthropic models (e.g., Claude-3)
- `"perplexity"`: Perplexity models

#### JSON Mode

Enable structured JSON responses:

```python
llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key="your-api-key",
    prompt="Format the data as JSON",
    json_mode=True,
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
)
```

#### Conversation History

Pass a list of messages to create a conversation:

```python
conversation = [
    "Can you help me understand quantum computing?",
    "Quantum computing uses quantum bits or qubits...",
    "How does that differ from classical computing?"
]

llm = SmartLLM(
    base="anthropic",
    model="claude-3-7-sonnet-20250219",
    api_key="your-api-key",
    prompt=conversation
)
```

### Provider-Specific Features

#### Anthropic

- Supports system prompts
- Full streaming support

#### OpenAI

- JSON schema validation
- Token counting

#### Perplexity

- Web search capabilities via search_recency_filter

### Caching

SmartLLM uses the Cacherator library for persistent JSON-based caching. LLM responses are automatically cached based on the configuration parameters, including the prompt, model, and other settings.

Key caching features:

- Results from both regular and streaming LLM requests are cached
- Cached results are stored in JSON format in the "data/llm" directory
- Cache files are named based on a unique identifier derived from the request parameters
- All properties (`content`, `json_content`, `sources`, `usage`) retrieve data from the cache when available

```python
# Configure cache TTL (time-to-live)
llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key="your-api-key",
    prompt="Your prompt",
    ttl=30  # Cache results for 30 days (default is 7 days)
)

# Force clear cache
llm = SmartLLM(
    base="openai",
    model="gpt-4",
    api_key="your-api-key",
    prompt="Your prompt",
    clear_cache=True  # Ignore existing cache
)
```

Internal methods decorated with `@Cached()` automatically manage caching of responses, using the cacherator library's persistence mechanism.

### Error Handling

SmartLLM provides robust error handling:

```python
llm = SmartLLM(...)
llm.generate()
llm.wait_for_completion()

if llm.is_failed():
    error = llm.get_error()
    print(f"Request failed: {error}")
else:
    print(llm._content)
```

### State Management

Track the state of requests:

```python
llm = SmartLLM(...)
llm.generate()

# Check state without blocking
if llm.is_pending():
    print("Request is still in progress")
elif llm.is_completed():
    print("Request completed successfully")
elif llm.is_failed():
    print(f"Request failed: {llm.get_error()}")
```

## Dependencies

- `cacherator`: Persistent JSON-based caching
- `logorator`: Decorator-based logging
- `openai>=1.0.0`: OpenAI API client
- `anthropic>=0.5.0`: Anthropic API client
- `python-slugify`: Utility for creating safe identifiers

## License

MIT License