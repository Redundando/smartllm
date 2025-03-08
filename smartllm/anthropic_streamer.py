from typing import Dict, List, Any, Callable, Optional
from anthropic import Anthropic
from logorator import Logger


class AnthropicStreamer:
    """
    Handles streaming functionality for the Anthropic API.
    This class encapsulates all streaming-related methods for Anthropic.
    """

    @Logger()
    def stream(
            self,
            client: Anthropic,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callback: Callable[[str], None]
    ) -> str:
        """
        Generate a streaming response from the Anthropic API.

        Args:
            client: Anthropic client
            model: Model name
            messages: List of message dictionaries
            params: Parameters for the API call
            callback: Function to call with each text chunk

        Returns:
            The complete text response
        """
        Logger.note(f"Sending streaming request to Anthropic API with model: {model}")

        # Remove the stream parameter from params if it exists
        # since stream() method already implies streaming
        if "stream" in params:
            del params["stream"]

        # Initialize full text response
        full_text = ""

        # Create a streaming response
        with client.messages.stream(**params) as stream:
            # Process each text chunk from the text_stream
            for text in stream.text_stream:
                if text:
                    full_text += text
                    callback(text)

        Logger.note("Completed streaming response from Anthropic API")
        return full_text

    def prepare_params(
            self,
            model: str,
            messages: List[Dict[str, str]],
            max_tokens: int,
            temperature: float,
            top_p: float,
            system_prompt: Optional[str] = None,
            json_mode: bool = False,
            json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare the parameters for a streaming request to the Anthropic API.

        Args:
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Temperature for response generation
            top_p: Top-p sampling value
            system_prompt: Optional system prompt
            json_mode: Whether to request JSON output
            json_schema: Schema for structured JSON output

        Returns:
            Dictionary of parameters for the Anthropic API
        """
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            # Anthropic doesn't use frequency_penalty or presence_penalty
        }

        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt

        # Add JSON support using Anthropic's tools feature
        if json_mode:
            # Create a tool definition for JSON output
            json_tool = {
                "name": "json_output",
                "description": "Output structured data in JSON format",
                "input_schema": json_schema or {"type": "object"}
            }
            params["tools"] = [json_tool]
            params["tool_choice"] = {"type": "tool", "name": "json_output"}

        return params

    def format_response(self, text: str) -> Dict[str, Any]:
        """
        Format the streamed text into a standardized response.
        This provides a minimal response structure for streaming.

        Args:
            text: The complete text from streaming

        Returns:
            Dictionary containing the response data
        """
        return {
            "content": text,
            "model": "streaming",  # Placeholder
            "id": "streaming",  # Placeholder
            "usage": {
                "prompt_tokens": 0,  # Not available in streaming
                "completion_tokens": 0,  # Not available in streaming
                "total_tokens": 0  # Not available in streaming
            }
        }