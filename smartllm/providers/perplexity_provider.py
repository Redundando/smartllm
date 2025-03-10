import json
from typing import Union, Optional, Dict, List, Any, Callable
from openai import OpenAI
from .base import LLMProvider
from logorator import Logger


class PerplexityProvider(LLMProvider):
    @Logger()
    def create_client(self, api_key: str, base_url: Optional[str] = None) -> OpenAI:
        Logger.note("Creating Perplexity API client")
        return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

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
        Logger.note(f"Perplexity streaming not yet implemented")
        raise NotImplementedError("Streaming not yet supported for Perplexity provider")

    def _configure_json_mode_with_schema(self, params: Dict[str, Any], json_schema: Dict[str, Any]) -> None:
        params["response_format"] = {
                "type"       : "json_schema",
                "json_schema": {"schema": json_schema or {"type": "object"}}
        }

    def extract_content(self, raw_response: Any) -> str:
        if hasattr(raw_response.choices[0], 'message') and hasattr(raw_response.choices[0].message, 'content'):
            return raw_response.choices[0].message.content
        return ""

    def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(raw_response.choices[0], 'message') and raw_response.choices[0].message.content:
                return json.loads(raw_response.choices[0].message.content)
            return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from Perplexity response: {str(e)}")
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