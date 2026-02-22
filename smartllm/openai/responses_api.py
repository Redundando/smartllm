"""OpenAI Response API implementation"""

import json
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
from logorator import Logger
from ..models import TextRequest, MessageRequest, TextResponse, StreamChunk
from ..utils import JSONFileCache


class ResponsesAPI:
    """Handler for OpenAI Response API"""
    
    def __init__(self, client, config, cache: JSONFileCache, semaphore=None):
        self.client = client
        self.config = config
        self.cache = cache
        self.semaphore = semaphore
    
    def __str__(self):
        return "OpenAI ResponsesAPI"

    @Logger(exclude_args=["invoke_with_retry"])
    async def generate_text(self, request: TextRequest, invoke_with_retry) -> TextResponse:
        """Generate text using Response API"""
        model = request.model or self.config.default_model
        is_reasoning = request.reasoning_effort is not None
        
        # Reasoning models don't support temperature
        if is_reasoning and request.temperature is not None and request.temperature != 1:
            raise ValueError(
                f"Reasoning models do not support temperature. "
                f"Remove temperature from your request or set it to 1."
            )
        
        temperature = None if is_reasoning else (request.temperature if request.temperature is not None else 0)
        
        # Cache key - reasoning models always cache (no temperature variation)
        cache_key = None
        if not request.stream and (temperature == 0 or is_reasoning):
            cache_key = self.cache._generate_key(
                api_type="responses",
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or self.config.max_tokens,
                instructions=request.system_prompt,
                reasoning_effort=request.reasoning_effort,
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
        Logger.note(f"{model} | reasoning={request.reasoning_effort or 'off'} | {prompt_preview}")
        
        # Build params
        params = {
            "model": model,
            "input": request.prompt,
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        
        if request.reasoning_effort:
            params["reasoning"] = {"effort": request.reasoning_effort}
        
        if request.system_prompt:
            params["instructions"] = request.system_prompt
        
        if request.max_tokens:
            params["max_output_tokens"] = request.max_tokens
        
        if request.top_p:
            params["top_p"] = request.top_p
        
        # Structured output
        if request.response_format:
            schema = request.response_format.model_json_schema()
            schema = self._clean_schema(schema)
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.response_format.__name__,
                    "schema": schema,
                    "strict": True
                }
            }
        
        try:
            if self.semaphore:
                async with self.semaphore:
                    response = await invoke_with_retry(self.client.responses.create, **params)
            else:
                response = await invoke_with_retry(self.client.responses.create, **params)
            
            result = self._parse_response(response, model, request.response_format)
            Logger.note(f"{result.input_tokens} in / {result.output_tokens} out | {result.text[:50]}")
            
            if cache_key:
                self.cache.set(cache_key, self._serialize_response(result), {"prompt": request.prompt})
            
            return result
        except Exception:
            raise
    
    def _clean_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively clean schema for OpenAI compatibility."""
        schema["additionalProperties"] = False
        
        if "properties" in schema:
            cleaned = {}
            for key, prop in schema["properties"].items():
                if "anyOf" in prop:
                    valid = [s for s in prop["anyOf"] if "type" in s or "$ref" in s]
                    if not valid:
                        continue
                    prop["anyOf"] = valid
                cleaned[key] = self._clean_schema(prop) if isinstance(prop, dict) else prop
            schema["properties"] = cleaned
            schema["required"] = list(cleaned.keys())
        
        if "$defs" in schema:
            schema["$defs"] = {k: self._clean_schema(v) for k, v in schema["$defs"].items()}
        
        return schema
    
    def _parse_response(self, response, model: str, response_format: Optional[Type[BaseModel]] = None) -> TextResponse:
        """Parse Response API response"""
        text = response.output_text or ""
        structured_data = None
        
        if response_format and text:
            try:
                data = json.loads(text)
                structured_data = response_format(**data)
            except Exception:
                pass
        
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        
        # Capture reasoning tokens in metadata if present
        metadata = {}
        if response.usage:
            reasoning_tokens = getattr(getattr(response.usage, "output_tokens_details", None), "reasoning_tokens", 0)
            cached_tokens = getattr(getattr(response.usage, "input_tokens_details", None), "cached_tokens", 0)
            if reasoning_tokens:
                metadata["reasoning_tokens"] = reasoning_tokens
            if cached_tokens:
                metadata["cached_tokens"] = cached_tokens
        
        return TextResponse(
            text=text,
            model=model,
            stop_reason=response.status or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata,
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
