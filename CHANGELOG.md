# Changelog

### Version 0.1.23
- **Bedrock — region resolution now matches boto3.** Resolution chain: explicit `aws_region=` arg → `AWS_REGION` → `AWS_DEFAULT_REGION` → package default. AWS environments (Lambda, ECS, EC2 with default profile) commonly set only `AWS_DEFAULT_REGION`, which 0.1.21/0.1.22 ignored — leading to surprising "model identifier is invalid" errors when consumers used cross-region inference profile IDs. Reported by package consumer.
- **Bedrock — defaults reverted to `us-east-1`.** `BEDROCK_DEFAULT_REGION = "us-east-1"`, `BEDROCK_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"`. The brief EU-default in 0.1.21/0.1.22 was a regression for the typical AWS deployment.
- **Bedrock — region hint on misleading errors.** When Bedrock returns "The provided model identifier is invalid" (which it does for both genuinely-invalid IDs and IDs that are valid but not deployed in the current region), `_maybe_log_region_hint` emits a one-line warning naming the resolved region and suggesting consumers check region/model alignment.
- **Bedrock — startup region log.** `_init_client` logs `Bedrock client initialized in region '<region>' (default model: <id>)` at info level so consumers can spot region/model mismatches up front.

### Version 0.1.22
- *(no functional change; metadata-only release)*

### Version 0.1.21
- **Bedrock — capability-aware request body construction.** Each Claude model on Bedrock now gets a request body that matches what it accepts. The new `smartllm.bedrock.capabilities` module is the single source of truth.
  - **Opus 4.7 / 4.8** — `temperature`, `top_p`, `top_k` are dropped (with a logged warning) instead of being sent and 400'd by Bedrock. Manual thinking budgets (`thinking.type=enabled`) are converted to adaptive thinking (`thinking.type=adaptive` + sibling `output_config.effort`).
  - **Opus 4.6** — also routed through adaptive thinking; sampling parameters are still accepted.
  - **Sonnet 4.6 and earlier** — unchanged behavior (manual thinking budget, full sampling parameters).
  - **Older / unknown Claude models** — permissive defaults (sampling accepted, no thinking).
- **Bedrock — `MessageRequest` parity.** `MessageRequest` now exposes `reasoning_effort`, `budget_tokens`, `top_p`, `top_k`. Multi-turn (`send_message`, `send_message_stream`) supports extended thinking on Claude.
- **Bedrock — public capability inspection.** `BedrockLLMClient.get_model_capabilities(model_id)` returns a `ModelCapabilities` dataclass; `BedrockLLMClient.supports_thinking(model_id)` is a shortcut. Also exported as `smartllm.bedrock.capabilities.get_model_capabilities` / `supports_thinking`.
- **Bedrock — tolerant tool-use parsing.** When structured output (`response_format`) lands as JSON-encoded strings inside the tool-use payload (observed on Sonnet 4.6 with non-English prompts), `_parse_response` retries with `json.loads` before raising. Pass-2 of the two-pass thinking + structure flow also instructs the model to return native arrays/objects.
- **Bedrock — defaults bumped.** `BEDROCK_DEFAULT_MODEL` is now `eu.anthropic.claude-sonnet-4-6`; `BEDROCK_DEFAULT_REGION` is now `eu-north-1`. `DEFAULT_MODEL_QUOTAS` recognises Claude 4.x families and Amazon Nova.
- **Tests — split conftest.** Unit tests in `tests/unit/` no longer prompt for a model. The `--model` picker moved to `tests/integration/conftest.py` and is TTY-aware (interactive in a real terminal, non-interactive elsewhere). Default model in `tests/models.toml` is now Bedrock-first.

**Backwards-compatibility notes (0.1.20 → 0.1.21):**
- `MessageRequest` and the new public methods are additive.
- `top_p` / `top_k` now reach modern Claude when explicitly set; previously they were silently dropped. On Sonnet 4.5 / Haiku 4.5 specifically, Bedrock rejects setting both `temperature` and `top_p` — narrow edge case.
- Cache keys for thinking-enabled requests use `request.budget_tokens` (often `None`) directly instead of the resolved integer. Existing cached entries won't hit on the new key — one-time cache rebuild, no correctness issue.
- Private method `BedrockLLMClient._resolve_thinking_budget` was removed (replaced by `_resolve_thinking_request` returning `bool`).

### Version 0.1.10
- Cache `data` no longer duplicates `prompt`/`messages`/`response_format` — these are stored only in the top-level `metadata`, reducing storage size

### Version 0.1.9
- *(no changes logged)*

### Version 0.1.8
- Fixed `BedrockLLMClient.__str__` to label the config default model as `default=` to avoid implying it reflects the per-request model
- Added `OpenAILLMClient.__str__` returning `OpenAILLMClient(default=<model>)` for consistency

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
