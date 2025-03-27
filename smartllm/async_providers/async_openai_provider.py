import json
import os
from typing import Callable, Optional

from openai import AsyncOpenAI

from .async_base import AsyncBaseProvider


def _get_api_key():
    try:
        return os.environ["CHATGPT_API_KEY"]
    except KeyError:
        return ""


class AsyncOpenAIProvider(AsyncBaseProvider):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompt = kwargs.get('prompt', "")
        self.system_prompt = kwargs.get('system_prompt')
        self.api_key = kwargs.get('api_key') or _get_api_key()
        self.model = kwargs.get('model', "")
        self.json_schema = kwargs.get('json_schema', None)
        self.stream = kwargs.get("stream") or  False
        self.callback: Optional[Callable[[str], None]] = kwargs.get("callback")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.response = None
        self._streaming_usage = None

    @property
    def _messages(self) -> list[dict]:
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        if isinstance(self.prompt, list):
            result += self.prompt
        if isinstance(self.prompt, str):
            result.append({"role": "user", "content": self.prompt})
        return result

    def _wrap_schema_for_openai(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "structured_response", "parameters": self.json_schema}}]

    async def execute(self):
        if self.stream:
            return await self._stream_response()

        kwargs = {"model": self.model, "messages": self._messages}

        if self.json_schema:
            kwargs["tools"] = self._wrap_schema_for_openai()
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "structured_response"}}

        self.response = await self.client.chat.completions.create(**kwargs)

        message = self.response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            return json.loads(message.tool_calls[0].function.arguments)

        return message.content

    async def _stream_response(self):
        if self.json_schema:
            raise ValueError("Streaming is not supported with tool calls / structured JSON output.")
        kwargs = {"model": self.model, "messages": self._messages, "stream": True, "stream_options": {"include_usage": True}}

        self.response = await self.client.chat.completions.create(**kwargs)

        full_response = ""

        async for chunk in self.response:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                full_response += delta
                if self.callback:
                    self.callback(delta)
            cm = chunk.model_dump()
            if "usage" in cm:
                self._streaming_usage = cm["usage"]

        return full_response

    async def models(self):
        models = await self.client.models.list()
        result = []
        for model in models.data:
            result.append(model.id)
        return result

    def usage(self):
        if self._streaming_usage:
            return self._streaming_usage
        usage = getattr(self.response, "usage", None)
        usage_dict = usage.model_dump()
        return usage_dict
