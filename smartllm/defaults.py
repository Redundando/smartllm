"""Default configuration values for SmartLLM

Users can modify these defaults by importing and changing them:

    from smartllm import defaults
    defaults.DEFAULT_TEMPERATURE = 0.7
    defaults.DEFAULT_MAX_TOKENS = 4096
"""

# Common defaults (shared across all providers)
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_MAX_RETRY_DELAY = 60.0

# Provider-specific defaults
# NOTE: Modern Anthropic models on Bedrock are inference-profile-only — pass an
# inference profile ID (e.g. `eu.…`, `us.…`, `global.…`) here, not a bare
# foundation model ID. The default below targets eu-north-1; switch to a
# `us.` / `global.` profile if running in a US region.
BEDROCK_DEFAULT_MODEL = "eu.anthropic.claude-sonnet-4-6"
BEDROCK_DEFAULT_REGION = "eu-north-1"
BEDROCK_DEFAULT_TOP_P = 0.9
BEDROCK_DEFAULT_TOP_K = 250

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_DEFAULT_TOP_P = 1.0

# Bedrock extended thinking budget mappings (reasoning_effort -> budget_tokens)
BEDROCK_THINKING_BUDGET = {
    "low": 1024,
    "medium": 4096,
    "high": 16000,
}
