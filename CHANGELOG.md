# Changelog

### Version 0.1.7
- `reasoning_tokens` and `cached_tokens` promoted to top-level `TextResponse` fields
- `timestamp` (ISO 8601 UTC) and `elapsed_seconds` added to `TextResponse`
- `metadata` now includes `prompt`/`messages` and `response_format` JSON schema on all live calls

### Version 0.1.6
- Added `on_progress` callback to `TextRequest` and `MessageRequest`
- Events: `llm_started`, `llm_done`, `cache_hit` (with `cache_source`), `error`
- Both sync and async callables supported
- `cache_source` on `TextResponse` indicates cache origin: `"miss"`, `"l1"`, or `"l2"`

### Version 0.1.5
- Replaced custom logging with [Logorator](https://pypi.org/project/logorator/) decorator-based logging
- Added two-level cache: local JSON files + optional DynamoDB via [Dynamorator](https://pypi.org/project/dynamorator/)
- DynamoDB cache configurable via `dynamo_table_name` and `cache_ttl_days` (default: 365 days)
- Cache write-back: DynamoDB hits are written to local cache automatically
- Prompt stored in cache metadata
- Recursive Pydantic schema cleaning for OpenAI structured output compatibility

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
- Unified interface for OpenAI and AWS Bedrock
- Async/await architecture
- Smart caching with temperature=0
- Auto retry with exponential backoff
- Structured output with Pydantic models
- Streaming responses
- Rate limiting and concurrency control
- OpenAI Response API support
- Reasoning model support with `reasoning_effort` parameter
