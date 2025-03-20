from typing import Union, Optional, Dict, List, Any, Callable
from anthropic import AsyncAnthropic
import json
from .base import AsyncLLMProvider
from logorator import Logger


class AsyncAnthropicProvider(AsyncLLMProvider):
    def __init__(self):
        self.system_prompt: Optional[str] = None
        self.json_mode: bool = False

    def __str__(self):
        return "AsyncAnthropicProvider"

    def __repr__(self):
        return self.__str__()

    async def create_client(self, api_key: str, base_url: Optional[str] = None,
                      api_version: Optional[str] = None) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=api_key)

    def _supports_system_prompt(self) -> bool:
        return False


    async def _execute_request(self, client: AsyncAnthropic, params: Dict[str, Any]) -> Any:
        return await client.messages.create(**params)

    async def _execute_streaming_request(
            self,
            client: AsyncAnthropic,
            params: Dict[str, Any],
            callbacks: List[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        content_buffer = ""
        stream = await client.messages.create(**params, stream=True)
        async for event in stream:
            chunk = ""
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                chunk = event.delta.text or ""
            elif hasattr(event, "delta") and hasattr(event.delta, "json"):
                partial_json_str = event.delta.model_dump_json() or "{}"
                partial_json = json.loads(partial_json_str)
                chunk = partial_json.get("partial_json") or ""
            content_buffer += chunk

            if callbacks and chunk:
                for callback in callbacks:
                    try:
                        callback(chunk, content_buffer)
                    except Exception as e:
                        Logger.note(f"Error in callback: {str(e)}")

        Logger.note(f"Streaming completed, total content length: {len(content_buffer)}")
        return await self.create_response_from_stream(content_buffer, params.get("model"), json_mode=self.json_mode)

    @Logger()
    async def generate_stream(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callbacks: List[Callable[[str, str], None]] = None,
    ) -> Any:
        Logger.note(f"Sending streaming request to Anthropic API with model: {model}")
        return await self._execute_streaming_request(client, params, callbacks)

    async def prepare_parameters(
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

    async def extract_content(self, raw_response: Any) -> str:
        content = ""
        for block in raw_response.content:
            if block.type == "text":
                content += block.text
        return content

    async def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(raw_response, 'content'):
                for block in raw_response.content:
                    if hasattr(block, 'type') and block.type == "tool_use" and block.name == "json_output":
                        return block.input
            return None
        except (AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from Anthropic response: {str(e)}")
            return None

    async def _extract_model_info(self, response: Any) -> str:
        return response.model

    async def _extract_response_id(self, response: Any) -> str:
        return response.id

    async def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        return {
                "prompt_tokens"    : response.usage.input_tokens,
                "completion_tokens": 0,  # Anthropic doesn't provide completion tokens
                "total_tokens"     : response.usage.input_tokens
        }

    @Logger()
    async def count_tokens(
            self,
            client: AsyncAnthropic,
            model: str,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None
    ) -> int:
        params = {"model": model, "messages": messages}

        if system_prompt:
            params["system"] = system_prompt

        response = await client.messages.count_tokens(**params)
        Logger.note(f"Token count: {response.input_tokens}")

        return response.input_tokens

    @Logger()
    async def list_models(
            self,
            client: AsyncAnthropic,
            limit: int = 20
    ) -> List[Dict[str, Any]]:

        response = await client.models.list(limit=limit)

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