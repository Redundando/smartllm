import json
from typing import Any, Dict, List, Optional, Union, Callable

from logorator import Logger
from openai import OpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    @Logger()
    def create_client(self, api_key: str, base_url: Optional[str] = None) -> OpenAI:
        Logger.note("Creating OpenAI API client")

        if not api_key:
            raise ValueError("OpenAI API key cannot be empty")

        client_args = {"api_key": api_key}

        if base_url:
            client_args["base_url"] = base_url

        return OpenAI(**client_args)

    def _execute_request(self, client: OpenAI, params: Dict[str, Any]) -> Any:
        return client.chat.completions.create(**params)

    @Logger()
    def generate_stream(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callbacks: List[Callable[[str, str], None]] = None,
    ) -> Any:
        Logger.note(f"OpenAI streaming not yet implemented")
        raise NotImplementedError("Streaming not yet supported for OpenAI provider")

    def _configure_json_mode_with_schema(self, params: Dict[str, Any], json_schema: Dict[str, Any]) -> None:
        params["tools"] = [{
                "type"    : "function",
                "function": {
                        "name"       : "json_output",
                        "description": "Structured JSON output",
                        "parameters" : json_schema
                }
        }]
        params["tool_choice"] = {"type": "function", "function": {"name": "json_output"}}

    def extract_content(self, raw_response: Any) -> str:
        if not hasattr(raw_response.choices[0], 'message'):
            return ""

        if not hasattr(raw_response.choices[0].message, 'content'):
            return ""

        return raw_response.choices[0].message.content

    def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(raw_response.choices[0], 'message') and hasattr(raw_response.choices[0].message, 'tool_calls'):
                for tool_call in raw_response.choices[0].message.tool_calls:
                    if tool_call.function.name == "json_output":
                        return json.loads(tool_call.function.arguments)

            if hasattr(raw_response.choices[0], 'message') and raw_response.choices[0].message.content:
                return json.loads(raw_response.choices[0].message.content)

            return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from OpenAI response: {str(e)}")
            return None

    def _extract_model_info(self, response: Any) -> str:
        return response.model

    def _extract_response_id(self, response: Any) -> str:
        return response.id

    def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        return {
                "prompt_tokens"    : response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens"     : response.usage.total_tokens
        }

    def count_tokens(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None
    ) -> int:
        from tiktoken import encoding_for_model

        try:
            encoding = encoding_for_model(model)
        except KeyError:
            encoding = encoding_for_model("gpt-3.5-turbo")

        token_count = 0

        for message in messages:
            token_count += 4
            for key, value in message.items():
                token_count += len(encoding.encode(value))
                if key == "name":
                    token_count += 1

        token_count += 3

        if system_prompt:
            token_count += 4
            token_count += len(encoding.encode(system_prompt))

        return token_count

    def list_models(
            self,
            client: Any,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        Logger.note("Listing available OpenAI models")

        response = client.models.list()

        models = [
                {
                        "id"        : model.id,
                        "name"      : model.id,
                        "created_at": model.created
                }
                for model in response.data[:limit]
        ]

        Logger.note(f"Found {len(models)} models")
        return models