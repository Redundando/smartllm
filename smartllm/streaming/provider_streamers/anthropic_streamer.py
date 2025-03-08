from typing import Dict, List, Any, Callable, Optional
from anthropic import Anthropic
from logorator import Logger


class AnthropicStreamer:
    def stream(
            self,
            client: Anthropic,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callback: Callable[[str], None]
    ) -> str:
        Logger.note(f"Sending streaming request to Anthropic API with model: {model}")

        if "stream" in params:
            del params["stream"]

        full_text = ""

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                if text:
                    full_text += text
                    callback(text)

        Logger.note("Completed streaming response from Anthropic API")
        return full_text

    def format_response(self, text: str) -> Dict[str, Any]:
        return {
            "content": text,
            "model": "streaming",
            "id": "streaming",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }