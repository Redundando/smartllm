from typing import Union, Optional, Dict, List, Any, Callable
from anthropic import Anthropic
from .base import LLMProvider
from logorator import Logger


class AnthropicProvider(LLMProvider):
    @Logger()
    def create_client(self, api_key: str, base_url: Optional[str] = None,
                      api_version: Optional[str] = None) -> Anthropic:
        Logger.note("Creating Anthropic API client")
        return Anthropic(api_key=api_key)

    def _execute_request(self, client: Anthropic, params: Dict[str, Any]) -> Any:
        """Execute request for Anthropic API"""
        return client.messages.create(**params)

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
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare parameters for Anthropic API (which has different parameter requirements)"""
        # Anthropic only supports certain parameters
        params = {
                "model"      : model,
                "messages"   : messages,
                "max_tokens" : max_tokens,
                "temperature": temperature,
                "top_p"      : top_p,
        }

        # Add system prompt for Anthropic (outside messages)
        if system_prompt:
            params["system"] = system_prompt

        # Add JSON mode configuration if requested
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
        """Extract content from Anthropic response"""
        content = ""
        for block in raw_response.content:
            if block.type == "text":
                content += block.text
        return content

    def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        """Extract JSON content from Anthropic response"""
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
        """Extract model information from Anthropic response"""
        return response.model

    def _extract_response_id(self, response: Any) -> str:
        """Extract response ID from Anthropic response"""
        return response.id

    def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        """Extract token usage information from Anthropic response"""
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