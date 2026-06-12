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
# inference profile ID (e.g. `us.…`, `eu.…`, `global.…`) here, not a bare
# foundation model ID. The defaults below match boto3's most common region
# (us-east-1) where Anthropic publishes the widest model coverage. Override
# via constructor args, BEDROCK_MODEL/AWS_REGION/AWS_DEFAULT_REGION env vars,
# or by mutating these constants.
BEDROCK_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
BEDROCK_DEFAULT_REGION = "us-east-1"
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
