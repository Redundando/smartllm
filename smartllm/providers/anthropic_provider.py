from typing import Union, Optional, Dict, List, Any, Callable
from anthropic import Anthropic
from httpx import stream

from .base import LLMProvider
from logorator import Logger


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.system_prompt: Optional[str] = None
        self.json_mode: bool = False

    @Logger()
    def create_client(self, api_key: str, base_url: Optional[str] = None,
                      api_version: Optional[str] = None) -> Anthropic:
        Logger.note("Creating Anthropic API client")
        return Anthropic(api_key=api_key)

    def _supports_system_prompt(self) -> bool:
        return False


    @Logger()
    def _execute_request(self, client: Anthropic, params: Dict[str, Any]) -> Any:
        return client.messages.create(**params)

    @Logger()
    def _execute_streaming_request(
            self,
            client: Anthropic,
            params: Dict[str, Any],
            callbacks: List[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        Logger.note(f"Executing Anthropic streaming request \n\n {params}")
        content_buffer = ""
        with client.messages.stream(**params) as stream:
            # This implementation is very ugly, but currently that's the only way to access partial chunks from Anthropic when streaming in JSON mode.
            stream_data = stream._raw_stream
            for text in stream_data:
                chunk = ""
                if hasattr(text, "delta") and hasattr(text.delta, "partial_json"):
                    chunk = str(text.delta.partial_json)
                if hasattr(text, "delta") and hasattr(text.delta, "text"):
                    chunk = str(text.delta.text)
                content_buffer += chunk
                if callbacks:
                    for callback in callbacks:
                        try:
                            callback(chunk, content_buffer)
                        except Exception as e:
                            Logger.note(f"Error in callback: {str(e)}")
        Logger.note(f"Streaming completed, total content length: {len(content_buffer)}")
        return self.create_response_from_stream(content_buffer, params.get("model"), json_mode=self.json_mode)

    @Logger()
    def generate_stream(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callbacks: List[Callable[[str, str], None]] = None,
    ) -> Any:
        Logger.note(f"Sending streaming request to Anthropic API with model: {model}")
        return self._execute_streaming_request(client, params, callbacks)

    def prepare_parameters(
            self,
            model: str,
            messages: List[Dict[str, str]],
            max_tokens: int,
            temperature: float,
            top_p: float,
            frequency_penalty: float,
            presence_penalty: float,
            search_recency_filter: Optional[str],
            json_mode: bool = False,
            json_schema: Optional[Dict[str, Any]] = None,
            system_prompt: Optional[str] = None,
            stream: bool = False,
    ) -> Dict[str, Any]:
        self.json_mode = json_mode
        self.system_prompt = system_prompt
        params = {
                "model"      : model,
                "messages"   : messages,
                "max_tokens" : max_tokens,
                "temperature": temperature,
                "top_p"      : top_p,
        }

        if system_prompt:
            params["system"] = system_prompt

        if json_mode and json_schema:
            json_tool = {
                    "name"        : "json_output",
                    "description" : "Output structured data in JSON format",
                    "input_schema": json_schema or {"type": "object"}
            }
            params["tools"] = [json_tool]
            params["tool_choice"] = {"type": "tool", "name": "json_output"}
        return params

    def extract_content(self, raw_response: Any) -> str:
        content = ""
        for block in raw_response.content:
            if block.type == "text":
                content += block.text
        return content

    def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(raw_response, 'content'):
                for block in raw_response.content:
                    if hasattr(block, 'type') and block.type == "tool_use" and block.name == "json_output":
                        return block.input
            return None
        except (AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from Anthropic response: {str(e)}")
            return None

    def _extract_model_info(self, response: Any) -> str:
        return response.model

    def _extract_response_id(self, response: Any) -> str:
        return response.id

    def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        return {
                "prompt_tokens"    : response.usage.input_tokens,
                "completion_tokens": 0,  # Anthropic doesn't provide completion tokens
                "total_tokens"     : response.usage.input_tokens
        }

    @Logger()
    def count_tokens(
            self,
            client: Anthropic,
            model: str,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None
    ) -> int:
        Logger.note(f"Counting tokens for model: {model}")

        params = {"model": model, "messages": messages}

        if system_prompt:
            params["system"] = system_prompt

        response = client.messages.count_tokens(**params)
        Logger.note(f"Token count: {response.input_tokens}")

        return response.input_tokens

    @Logger()
    def list_models(
            self,
            client: Anthropic,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        Logger.note("Listing available Anthropic models")

        response = client.models.list(limit=limit)

        models = [
                {
                        "id"        : model.id,
                        "name"      : model.display_name,
                        "created_at": model.created_at
                }
                for model in response.data
        ]

        Logger.note(f"Found {len(models)} models")
        return models