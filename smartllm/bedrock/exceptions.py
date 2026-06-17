"""Custom exceptions for the Bedrock provider.

Hierarchy
---------

    Exception
      └── BedrockError                    (base for all Bedrock-specific errors)
            ├── BedrockStreamError        (stream-level error event from Bedrock)
            └── BedrockStreamTimeoutError (no first chunk / overall stall)

All Bedrock-specific exceptions inherit from `BedrockError` so consumers
can catch broadly with one `except`. Specific subclasses carry structured
context so callers can distinguish failure modes (throttling vs.
validation vs. timeout) without string-parsing.
"""

from typing import Any, Dict, Optional


# Stream-level error event keys delivered by Bedrock's response stream.
# These are NOT wrapped as `chunk` events — they arrive as separate top-level
# keys and must be detected explicitly. Documented at:
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ResponseStream.html
STREAM_ERROR_EVENT_KEYS = (
    "internalServerException",
    "modelStreamErrorException",
    "validationException",
    "throttlingException",
    "serviceUnavailableException",
    "modelTimeoutException",
)

# Subset of the above that should participate in the existing retry loop
# (transient / server-side / capacity errors). Validation errors are not
# retryable — the request shape is wrong and won't fix itself.
RETRYABLE_STREAM_ERROR_TYPES = frozenset({
    "throttlingException",
    "serviceUnavailableException",
    "modelTimeoutException",
    "internalServerException",
})


class BedrockError(Exception):
    """Base class for all smartllm Bedrock-specific exceptions."""


class BedrockStreamError(BedrockError):
    """A Bedrock stream delivered an error event instead of (or in addition
    to) chunks.

    These events are emitted by Bedrock's `InvokeModelWithResponseStream` API
    as separate top-level keys (e.g. `throttlingException`,
    `modelStreamErrorException`). Prior to this exception class, smartllm
    silently skipped them and returned an empty/partial response.

    Attributes:
        error_type: The Bedrock event key (one of `STREAM_ERROR_EVENT_KEYS`).
        message: The human-readable message from the event payload, if any.
        raw: The full raw event dict as received from boto3, useful for
            debugging unknown payload shapes.
    """

    def __init__(
        self,
        error_type: str,
        message: str = "",
        raw: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.raw = raw or {}
        super().__init__(
            f"Bedrock stream error event '{error_type}': {message}"
            if message
            else f"Bedrock stream error event '{error_type}'"
        )

    @property
    def is_retryable(self) -> bool:
        """Whether this error type is in the transient/retryable category."""
        return self.error_type in RETRYABLE_STREAM_ERROR_TYPES


class BedrockStreamTimeoutError(BedrockError):
    """A Bedrock streaming connection stalled past a configured timeout.

    Two flavors are reported via the `kind` attribute:
      - "first_chunk": no event arrived within `stream_first_chunk_timeout`
        seconds of the request being accepted (typically queueing under TPM
        saturation).
      - "total": the stream did not finish within `stream_total_timeout`
        seconds (typically a half-closed connection or a runaway generation).

    Timeouts are NOT considered retryable by default — they indicate a
    sustained problem rather than a transient one. Callers can implement
    their own retry policy if appropriate.

    Attributes:
        kind: Either "first_chunk" or "total".
        elapsed: Seconds elapsed when the timeout fired.
    """

    def __init__(self, kind: str, elapsed: float):
        if kind not in ("first_chunk", "total"):
            raise ValueError(
                f"BedrockStreamTimeoutError.kind must be 'first_chunk' or 'total', "
                f"got '{kind}'"
            )
        self.kind = kind
        self.elapsed = elapsed
        super().__init__(
            f"Bedrock stream {kind} timeout after {elapsed:.1f}s"
        )

    @property
    def is_retryable(self) -> bool:
        """Stream timeouts are not retryable by default."""
        return False
