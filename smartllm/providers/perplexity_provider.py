import json
from typing import Union, Optional, Dict, List, Any
from openai import OpenAI
from .base import LLMProvider
from logorator import Logger


class PerplexityProvider(LLMProvider):
    @Logger()
    def create_client(self, api_key: str, base_url: Optional[str] = None) -> OpenAI:
        Logger.note("Creating Perplexity API client")
        return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    def _execute_request(self, client: OpenAI, params: Dict[str, Any]) -> Any:
        """Execute request for Perplexity API"""
        return client.chat.completions.create(**params)

    def _configure_json_mode_with_schema(self, params: Dict[str, Any], json_schema: Dict[str, Any]) -> None:
        """Configure JSON mode with schema for Perplexity API"""
        params["response_format"] = {
                "type"       : "json_schema",
                "json_schema": {"schema": json_schema or {"type": "object"}}
        }

    def extract_content(self, raw_response: Any) -> str:
        """Extract content from Perplexity response"""
        if hasattr(raw_response.choices[0], 'message') and hasattr(raw_response.choices[0].message, 'content'):
            return raw_response.choices[0].message.content
        return ""

    def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        """Extract JSON content from Perplexity response"""
        try:
            if hasattr(raw_response.choices[0], 'message') and raw_response.choices[0].message.content:
                return json.loads(raw_response.choices[0].message.content)
            return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from Perplexity response: {str(e)}")
            return None

    def _extract_model_info(self, response: Any) -> str:
        """Extract model information from Perplexity response"""
        return response.model

    def _extract_response_id(self, response: Any) -> str:
        """Extract response ID from Perplexity response"""
        return response.id

    def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        """Extract token usage information from Perplexity response"""
        return {
                "prompt_tokens"    : response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens"     : response.usage.total_tokens
        }