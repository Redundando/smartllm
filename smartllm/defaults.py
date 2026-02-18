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
BEDROCK_DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_DEFAULT_REGION = "us-east-1"
BEDROCK_DEFAULT_TOP_P = 0.9
BEDROCK_DEFAULT_TOP_K = 250

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_DEFAULT_TOP_P = 1.0
