"""Per-model capability detection for Bedrock-hosted Anthropic Claude models.

Modern Claude models on Bedrock have diverged in how they accept request
parameters. Most notably, Opus 4.6+ rejects manual extended-thinking budgets
and requires the adaptive-thinking shape, and Opus 4.7/4.8 also reject the
classical sampling parameters (temperature, top_p, top_k).

This module provides one source of truth so that body-construction code can
ask "what does this model accept?" instead of hardcoding shapes.

Matching is by lowercase substring on the model ID, which means it works for
bare foundation IDs (`anthropic.claude-opus-4-7`) and inference profile IDs
with region prefixes (`eu.anthropic.…`, `us.…`, `global.…`).

References:
    - Anthropic prompt validation: temperature/top_p/top_k deprecated on Opus 4.7+
      https://docs.anthropic.com/en/api/prompt-validation
    - Bedrock model card for Opus 4.7: thinking.type=adaptive only
      https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html
"""

from dataclasses import dataclass
from typing import Literal


ThinkingMode = Literal["none", "manual_budget", "adaptive_effort"]

# Effort levels accepted by adaptive thinking. Anthropic exposes more values
# (xhigh, max) on some plans; we only support the three our public
# `reasoning_effort` enum uses, since callers map through that.
ADAPTIVE_EFFORT_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class ModelCapabilities:
    """What a Claude model on Bedrock will accept in the request body."""
    family: str
    accepts_temperature: bool
    accepts_top_p_top_k: bool
    thinking_mode: ThinkingMode


# Order matters: more specific patterns must come before more general ones,
# because matching is first-substring-wins. (e.g. "claude-opus-4-7" must be
# checked before "claude-opus-4".)
_CAPABILITY_RULES: tuple[tuple[str, ModelCapabilities], ...] = (
    # Opus 4.6+ — adaptive thinking only; 4.7+ also drops sampling params
    ("claude-opus-4-8", ModelCapabilities("claude-opus-4-8", False, False, "adaptive_effort")),
    ("claude-opus-4-7", ModelCapabilities("claude-opus-4-7", False, False, "adaptive_effort")),
    ("claude-opus-4-6", ModelCapabilities("claude-opus-4-6", True,  True,  "adaptive_effort")),
    # Sonnet/Opus 4.x — manual budget thinking, full sampling params
    ("claude-sonnet-4-6", ModelCapabilities("claude-sonnet-4-6", True, True, "manual_budget")),
    ("claude-sonnet-4-5", ModelCapabilities("claude-sonnet-4-5", True, True, "manual_budget")),
    ("claude-opus-4-5",   ModelCapabilities("claude-opus-4-5",   True, True, "manual_budget")),
    ("claude-sonnet-4",   ModelCapabilities("claude-sonnet-4",   True, True, "manual_budget")),
    ("claude-opus-4",     ModelCapabilities("claude-opus-4",     True, True, "manual_budget")),
    # Claude 3.7 — thinking introduced here
    ("claude-3-7-sonnet", ModelCapabilities("claude-3-7-sonnet", True, True, "manual_budget")),
    # Claude 3.x — no thinking
    ("claude-3-5-sonnet", ModelCapabilities("claude-3-5-sonnet", True, True, "none")),
    ("claude-3-5-haiku",  ModelCapabilities("claude-3-5-haiku",  True, True, "none")),
    ("claude-3-sonnet",   ModelCapabilities("claude-3-sonnet",   True, True, "none")),
    ("claude-3-haiku",    ModelCapabilities("claude-3-haiku",    True, True, "none")),
    ("claude-3-opus",     ModelCapabilities("claude-3-opus",     True, True, "none")),
)

# Permissive default for unknown Claude models: assume the model accepts
# everything the older API accepted, but no thinking. Logged once per ID
# at the call site.
_UNKNOWN_CLAUDE = ModelCapabilities("claude-unknown", True, True, "none")


def get_model_capabilities(model_id: str) -> ModelCapabilities:
    """Return capabilities for a Claude model ID.

    For non-Claude models or unrecognised Claude variants, returns a permissive
    default that mirrors pre-thinking Claude behaviour.
    """
    needle = model_id.lower()
    for pattern, caps in _CAPABILITY_RULES:
        if pattern in needle:
            return caps
    return _UNKNOWN_CLAUDE


def supports_thinking(model_id: str) -> bool:
    """Whether the model supports any form of extended thinking."""
    return get_model_capabilities(model_id).thinking_mode != "none"
