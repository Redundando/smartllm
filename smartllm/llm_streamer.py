from typing import Dict, List, Any, Callable, Optional, Union
from enum import Enum
from logorator import Logger


class LLMRequestState(Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class SmartLLMStreamer:
    """
    Handles streaming functionality for the SmartLLM class.
    This class encapsulates all streaming-related methods.
    """

    def __init__(self):
        self.state = LLMRequestState.NOT_STARTED
        self.error = None
        self._generation_result = None

    @Logger()
    def generate(
            self,
            base: str,
            provider,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate a streaming response from the LLM.

        Args:
            base: Provider name (e.g., "anthropic")
            provider: Provider instance
            client: API client
            model: Model name
            messages: List of message dictionaries
            params: Parameters for the API call
            callback: Function to call with each text chunk (optional)

        Returns:
            Dictionary containing the response data
        """
        Logger.note(f"Starting streaming LLM request for {base}/{model}")

        if self.state == LLMRequestState.PENDING:
            Logger.note("Request already in progress, not starting a new one")
            return self._generation_result

        if self.state == LLMRequestState.COMPLETED:
            Logger.note("Request already completed, not starting a new one")
            return self._generation_result

        # Use the default callback if none is provided
        if callback is None:
            callback = self.handle_chunk

        self.state = LLMRequestState.PENDING

        try:
            # Generate streaming response based on provider
            full_text = None

            if base == "anthropic":
                from .anthropic_streamer import AnthropicStreamer
                streamer = AnthropicStreamer()
                full_text = streamer.stream(
                    client=client,
                    model=model,
                    messages=messages,
                    params=params,
                    callback=callback
                )
                self._generation_result = streamer.format_response(full_text)
            elif base == "openai":
                # OpenAI streaming implementation would go here
                Logger.note("OpenAI streaming not yet implemented")
                raise NotImplementedError("OpenAI streaming not yet implemented")
            elif base == "perplexity":
                # Perplexity streaming implementation would go here
                Logger.note("Perplexity streaming not yet implemented")
                raise NotImplementedError("Perplexity streaming not yet implemented")
            else:
                raise ValueError(f"Streaming not supported for provider: {base}")

            self.state = LLMRequestState.COMPLETED
            Logger.note(f"Streaming LLM request completed successfully")

            return self._generation_result

        except Exception as e:
            self.state = LLMRequestState.FAILED
            self.error = str(e)
            Logger.note(f"Streaming LLM request failed: {str(e)}")
            raise

    def handle_chunk(self, chunk: str) -> None:
        """
        Default callback function for handling streaming chunks.
        Simply logs the chunk.

        Args:
            chunk: Text chunk from the streaming response
        """
        Logger.note(f"Received chunk: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")

    def is_completed(self) -> bool:
        """
        Check if the streaming request has completed successfully.

        Returns:
            True if completed, False otherwise
        """
        return self.state == LLMRequestState.COMPLETED

    def is_pending(self) -> bool:
        """
        Check if the streaming request is still in progress.

        Returns:
            True if pending, False otherwise
        """
        return self.state == LLMRequestState.PENDING

    def is_failed(self) -> bool:
        """
        Check if the streaming request has failed.

        Returns:
            True if failed, False otherwise
        """
        return self.state == LLMRequestState.FAILED

    def get_error(self) -> Optional[str]:
        """
        Get the error message if the streaming request failed.

        Returns:
            Error message or None if no error
        """
        return self.error if self.is_failed() else None