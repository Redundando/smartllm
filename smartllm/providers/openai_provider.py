import json
from typing import Any, Dict, List, Optional, Union

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

    @Logger()
    def generate(
            self,
            client: OpenAI,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
    ) -> Any:
        Logger.note(f"Sending request to OpenAI API with model: {model}")
        response = client.chat.completions.create(**params)
        Logger.note("Received response from OpenAI API")
        return response

    def prepare_messages(
            self,
            prompt: Union[str, List[str]],
            system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

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
            system_prompt: Optional[str] = None) -> Dict[str, Any]:
        params = {
                "model"            : model,
                "messages"         : messages,
                "temperature"      : temperature,
                "top_p"            : top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty" : presence_penalty,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if search_recency_filter and search_recency_filter in ["month", "week", "day", "hour"]:
            params["search_recency_filter"] = search_recency_filter

        if json_mode:
            if json_schema:
                params["tools"] = [{
                        "type"    : "function",
                        "function": {
                                "name"       : "json_output",
                                "description": "Structured JSON output",
                                "parameters" : json_schema
                        }
                }]
                params["tool_choice"] = {"type": "function", "function": {"name": "json_output"}}
            else:
                params["response_format"] = {"type": "json_object"}

        return params

    def format_response(
            self,
            response: Any,
            return_citations: bool
    ) -> Dict[str, Any]:
        result = {
                "content"     : self.extract_content(response),
                "model"       : response.model,
                "id"          : response.id,
                "usage"       : {
                        "prompt_tokens"    : response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens"     : response.usage.total_tokens
                },
                "raw_response": response
        }

        if return_citations and hasattr(response, 'citations'):
            result["citations"] = response.citations

        return result

    def format_json_response(
            self,
            response: Any
    ) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'tool_calls'):
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "json_output":
                        return json.loads(tool_call.function.arguments)

            if hasattr(response.choices[0], 'message') and response.choices[0].message.content:
                return json.loads(response.choices[0].message.content)

            return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from OpenAI response: {str(e)}")
            return None

    def extract_content(self, raw_response: Any) -> str:
        if not hasattr(raw_response.choices[0], 'message'):
            return ""

        if not hasattr(raw_response.choices[0].message, 'content'):
            return ""

        return raw_response.choices[0].message.content

    def create_serializable_response(
            self,
            raw_response: Any,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        content = self.extract_content(raw_response)

        citations = []
        if hasattr(raw_response, 'citations'):
            citations = raw_response.citations

        serializable = {
                "content"  : content,
                "model"    : raw_response.model,
                "id"       : raw_response.id,
                "usage"    : {
                        "prompt_tokens"    : raw_response.usage.prompt_tokens,
                        "completion_tokens": raw_response.usage.completion_tokens,
                        "total_tokens"     : raw_response.usage.total_tokens
                },
                "citations": citations
        }

        if json_mode:
            json_content = self.format_json_response(raw_response)
            if json_content:
                serializable["json_content"] = json_content

        return serializable

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
