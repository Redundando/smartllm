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

    @Logger()
    def generate(
            self,
            client: Anthropic,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
    ) -> Any:
        Logger.note(f"Sending request to Anthropic API with model: {model}")
        response = client.messages.create(**params)
        Logger.note("Received response from Anthropic API")
        return response

    def prepare_messages(
            self,
            prompt: Union[str, List[str]],
            system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages = []

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            for i, msg in enumerate(prompt):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})

        return messages

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
            system_prompt: Optional[str] = None    ) -> Dict[str, Any]:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }


        if system_prompt:
            params["system"] = system_prompt

        if json_mode:
            json_tool = {
                "name": "json_output",
                "description": "Output structured data in JSON format",
                "input_schema": json_schema or {"type": "object"}
            }
            params["tools"] = [json_tool]
            params["tool_choice"] = {"type": "tool", "name": "json_output"}

        return params

    def format_response(
            self,
            response: Any,
            return_citations: bool
    ) -> Dict[str, Any]:
        content = self.extract_content(response)

        result = {
            "content": content,
            "model": response.model,
            "id": response.id,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": 0,
                "total_tokens": response.usage.input_tokens
            },
            "raw_response": response
        }

        return result

    def format_json_response(
            self,
            response: Any
    ) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(response, 'content'):
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "tool_use" and block.name == "json_output":
                        return block.input
            return None
        except (AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from Anthropic response: {str(e)}")
            return None

    def extract_content(self, raw_response: Any) -> str:
        content = ""
        for block in raw_response.content:
            if block.type == "text":
                content += block.text
        return content

    def create_serializable_response(
            self,
            raw_response: Any,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        content = self.extract_content(raw_response)

        citations = []

        serializable = {
            "content": content,
            "model": raw_response.model,
            "id": raw_response.id,
            "usage": {
                "prompt_tokens": raw_response.usage.input_tokens,
                "completion_tokens": 0,
                "total_tokens": raw_response.usage.input_tokens
            },
            "citations": citations
        }

        if json_mode:
            json_content = self.format_json_response(raw_response)
            if json_content:
                serializable["json_content"] = json_content

        return serializable

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
                "id": model.id,
                "name": model.display_name,
                "created_at": model.created_at
            }
            for model in response.data
        ]

        Logger.note(f"Found {len(models)} models")
        return models