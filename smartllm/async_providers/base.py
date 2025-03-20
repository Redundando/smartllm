from typing import Union, Optional, Dict, List, Any, Callable
from logorator import Logger


class AsyncLLMProvider:
    async def create_client(self, api_key: str, base_url: Optional[str] = None) -> Any:
        raise NotImplementedError("Subclasses must implement create_client")

    def __str__(self):
        return "AsyncLLMProvider"

    def __repr__(self):
        return "AsyncLLMProvider"

    @Logger(override_function_name="LLM API call")
    async def generate(
            self,
            client: Any,
            params: Dict[str, Any],
    ) -> Any:
        response = await self._execute_request(client, params)
        return response

    @Logger()
    async def generate_stream(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
            callbacks: List[Callable[[str, str], None]] = None,
    ) -> Any:
        Logger.note(f"Sending streaming request to {self.__class__.__name__} API with model: {model}")
        raise NotImplementedError(f"Streaming not supported by {self.__class__.__name__}")

    async def _execute_request(self, client: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement _execute_request")

    async def _execute_streaming_request(self, client: Any, params: Dict[str, Any], callbacks: List[Callable] = None) -> Any:
        raise NotImplementedError(f"Streaming not supported by {self.__class__.__name__}")

    async def prepare_messages(
            self,
            prompt: Union[str, List[str]],
            system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages = []

        if system_prompt and self._supports_system_prompt():
            messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            for i, msg in enumerate(prompt):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})

        return messages

    async def _supports_system_prompt(self) -> bool:
        return True

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
            stream: bool = False
    ) -> Dict[str, Any]:
        params = {
                "model"      : model,
                "messages"   : messages,
                "temperature": temperature,
                "top_p"      : top_p
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        params["frequency_penalty"] = frequency_penalty
        params["presence_penalty"] = presence_penalty

        if search_recency_filter and search_recency_filter in ["month", "week", "day", "hour"]:
            params["search_recency_filter"] = search_recency_filter

        if json_mode:
            if json_schema:
                await self._configure_json_mode_with_schema(params, json_schema)
            else:
                params["response_format"] = {"type": "json_object"}

        if system_prompt and not self._supports_system_prompt():
            params["system"] = system_prompt

        if stream:
            params["stream"] = stream

        return params

    async  def _configure_json_mode_with_schema(self, params: Dict[str, Any], json_schema: Dict[str, Any]) -> None:
        params["response_format"] = {"type": "json_object"}

    async  def extract_content(self, raw_response: Any) -> str:
        raise NotImplementedError("Subclasses must implement extract_content")

    async  def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        return None

    async def create_response(
            self,
            raw_response: Any,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        content = await self.extract_content(raw_response)

        model = await self._extract_model_info(raw_response)
        response_id = await self._extract_response_id(raw_response)
        usage = await self._extract_usage_info(raw_response)

        response = {
                "content"  : content,
                "model"    : model,
                "id"       : response_id,
                "usage"    : usage,
                "citations": await self._extract_citations(raw_response)
        }

        if json_mode:
            json_content = await self.extract_json_content(raw_response)
            if json_content:
                response["json_content"] = json_content

        return response

    async def create_response_from_stream(
            self,
            content: str,
            model: str,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        response_id = f"stream_{model}_{hash(content)}"

        # Create a default usage object - actual token counting would need
        # to be implemented provider-specific
        usage = {
                "input_tokens" : 0,
                "output_tokens": 0,
                "total_tokens" : 0
        }

        response = {
                "content"  : content,
                "model"    : model,
                "id"       : response_id,
                "usage"    : usage,
                "citations": []
        }

        # Handle JSON content if needed
        if json_mode and content:
            try:
                import json
                json_content = json.loads(content)
                response["json_content"] = json_content
            except Exception:
                pass
        return response

    async def _extract_model_info(self, response: Any) -> str:
        raise NotImplementedError("Subclasses must implement _extract_model_info")

    async def _extract_response_id(self, response: Any) -> str:
        raise NotImplementedError("Subclasses must implement _extract_response_id")

    async def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        raise NotImplementedError("Subclasses must implement _extract_usage_info")

    async def _extract_citations(self, response: Any) -> List[str]:
        return getattr(response, 'citations', [])