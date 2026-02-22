"""OpenAI Chat Completions API implementation"""

import json
from typing import Optional, Type, Dict, Any, AsyncIterator
from pydantic import BaseModel
from logorator import Logger
from ..models import TextRequest, MessageRequest, TextResponse, StreamChunk
from ..utils import pydantic_to_tool_schema, JSONFileCache


class ChatCompletionsAPI:
    """Handler for OpenAI Chat Completions API"""
    
    def __init__(self, client, config, cache: JSONFileCache, semaphore=None):
        self.client = client
        self.config = config
        self.cache = cache
        self.semaphore = semaphore
    
    def __str__(self):
        return "OpenAI ChatCompletionsAPI"

    @Logger(exclude_args=["invoke_with_retry"])
    async def generate_text(self, request: TextRequest, invoke_with_retry) -> TextResponse:
        """Generate text using Chat Completions API"""
        model = request.model or self.config.default_model
        temperature = request.temperature if request.temperature is not None else 0
        
        # Cache key
        cache_key = None
        if temperature == 0 and not request.stream:
            cache_key = self.cache._generate_key(
                api_type="chat_completions",
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None
            )
        
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
        
        if request.use_cache and cache_key:
            cached, cache_source = self.cache.get(cache_key)
            if cached:
                Logger.note(f"Cache hit [{cache_key[:8]}] - {model}")
                result = self._deserialize_response(cached["data"], request.response_format)
                result.cache_source = cache_source
                return result
        
        prompt_preview = request.prompt[:60] + "..." if len(request.prompt) > 60 else request.prompt
        Logger.note(f"{model} | temp={temperature} | {prompt_preview}")
        
        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        # Build params
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "top_p": request.top_p or self.config.top_p,
        }
        
        # Structured output
        if request.response_format:
            params["tools"] = [self._build_tool_schema(request.response_format)]
            params["tool_choice"] = {"type": "function", "function": {"name": params["tools"][0]["function"]["name"]}}
        
        try:
            if self.semaphore:
                async with self.semaphore:
                    response = await invoke_with_retry(self.client.chat.completions.create, **params)
            else:
                response = await invoke_with_retry(self.client.chat.completions.create, **params)
            
            result = self._parse_response(response, model, request.response_format)
            Logger.note(f"{result.input_tokens} in / {result.output_tokens} out | {result.text[:50]}")
            
            if cache_key:
                self.cache.set(cache_key, self._serialize_response(result), {})
            
            return result
        except Exception:
            raise
    
    async def generate_text_stream(self, request: TextRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation"""
        model = request.model or self.config.default_model
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature or self.config.temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "stream": True,
        }
        
        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamChunk(text=chunk.choices[0].delta.content, model=model)
        except Exception:
            raise
    
    @Logger(exclude_args=["invoke_with_retry"])
    async def send_message(self, request: MessageRequest, invoke_with_retry) -> TextResponse:
        """Send a message in a conversation"""
        model = request.model or self.config.default_model
        temperature = request.temperature if request.temperature is not None else 0
        
        # Cache key
        cache_key = None
        if temperature == 0 and not request.stream:
            messages_str = json.dumps([{"role": m.role, "content": m.content} for m in request.messages])
            cache_key = self.cache._generate_key(
                api_type="chat_completions",
                model=model,
                messages=messages_str,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system_prompt=request.system_prompt,
                response_format=request.response_format.__name__ if request.response_format else None
            )
        
        if request.clear_cache and cache_key:
            self.cache.clear(cache_key)
        
        if request.use_cache and cache_key:
            cached, cache_source = self.cache.get(cache_key)
            if cached:
                Logger.note(f"Cache hit [{cache_key[:8]}] - {model}")
                result = self._deserialize_response(cached["data"], request.response_format)
                result.cache_source = cache_source
                return result
        
        last_msg = request.messages[-1].content[:60] if request.messages else ""
        Logger.note(f"{model} | {len(request.messages)} messages | {last_msg}")
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
        }
        
        if request.response_format:
            params["tools"] = [self._build_tool_schema(request.response_format)]
            params["tool_choice"] = {"type": "function", "function": {"name": params["tools"][0]["function"]["name"]}}
        
        try:
            if self.semaphore:
                async with self.semaphore:
                    response = await invoke_with_retry(self.client.chat.completions.create, **params)
            else:
                response = await invoke_with_retry(self.client.chat.completions.create, **params)
            
            result = self._parse_response(response, model, request.response_format)
            Logger.note(f"{result.input_tokens} in / {result.output_tokens} out | {result.text[:50]}")
            
            if cache_key:
                self.cache.set(cache_key, self._serialize_response(result), {})
            
            return result
        except Exception:
            raise
    
    async def send_message_stream(self, request: MessageRequest) -> AsyncIterator[StreamChunk]:
        """Stream a conversation message"""
        model = request.model or self.config.default_model
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature or self.config.temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "stream": True,
        }
        
        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamChunk(text=chunk.choices[0].delta.content, model=model)
        except Exception:
            raise
    
    def _build_tool_schema(self, response_format: Type[BaseModel]) -> Dict[str, Any]:
        """Build OpenAI tool schema from Pydantic model"""
        schema = pydantic_to_tool_schema(response_format)
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"]
            }
        }
    
    def _parse_response(self, response, model: str, response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Parse Chat Completions response"""
        choice = response.choices[0]
        
        # Check for tool calls (structured output)
        if choice.message.tool_calls and response_format:
            tool_call = choice.message.tool_calls[0]
            tool_input = json.loads(tool_call.function.arguments)
            structured_data = response_format(**tool_input)
            text = json.dumps(tool_input, indent=2)
        else:
            text = choice.message.content or ""
            structured_data = None
        
        return TextResponse(
            text=text,
            model=model,
            stop_reason=choice.finish_reason or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            structured_data=structured_data,
        )
    
    def _serialize_response(self, response: TextResponse) -> Dict[str, Any]:
        """Serialize TextResponse for caching"""
        return {
            "text": response.text,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "metadata": response.metadata,
            "structured_data": response.structured_data.model_dump() if response.structured_data else None,
        }
    
    def _deserialize_response(self, data: Dict[str, Any], response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Deserialize cached data back to TextResponse"""
        structured_data = None
        if data.get("structured_data") and response_format:
            structured_data = response_format(**data["structured_data"])
        
        return TextResponse(
            text=data["text"],
            model=data["model"],
            stop_reason=data["stop_reason"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            metadata=data.get("metadata", {}),
            structured_data=structured_data,
        )
