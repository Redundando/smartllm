from typing import Union, Optional, Dict, List, Any
from openai import OpenAI
from .base import LLMProvider
from logorator import Logger
import json


class PerplexityProvider(LLMProvider):
    @Logger()
    def create_client(self, api_key: str, base_url: Optional[str] = None) -> OpenAI:
        Logger.note("Creating Perplexity API client")
        return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    @Logger()
    def generate(
            self,
            client: OpenAI,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
    ) -> Any:
        Logger.note(f"Sending request to Perplexity API with model: {model}")
        response = client.chat.completions.create(**params)
        Logger.note("Received response from Perplexity API")
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
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        if search_recency_filter and search_recency_filter in ["month", "week", "day", "hour"]:
            params["search_recency_filter"] = search_recency_filter

        if json_mode:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": json_schema or {"type": "object"}}
            }

        return params

    def format_response(
            self,
            response: Any,
            return_citations: bool
    ) -> Dict[str, Any]:
        result = {
            "content": self.extract_content(response),
            "model": response.model,
            "id": response.id,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
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
            if hasattr(response.choices[0], 'message') and response.choices[0].message.content:
                return json.loads(response.choices[0].message.content)
            return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            Logger.note(f"Error extracting JSON from Perplexity response: {str(e)}")
            return None

    def extract_content(self, raw_response: Any) -> str:
        if hasattr(raw_response.choices[0], 'message') and hasattr(raw_response.choices[0].message, 'content'):
            return raw_response.choices[0].message.content
        return ""

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
            "content": content,
            "model": raw_response.model,
            "id": raw_response.id,
            "usage": {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
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
        raise NotImplementedError("Token counting not implemented for Perplexity")

    def list_models(
            self,
            client: Any,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Listing models not implemented for Perplexity")