from typing import Union, Optional, Dict, List, Any
from hashlib import sha256
import json


class Configuration:
    def __init__(
            self,
            base: str = "",
            model: str = "",
            api_key: str = "",
            prompt: Union[str, List[str]] = "",
            max_input_tokens: Optional[int] = None,
            max_output_tokens: Optional[int] = None,
            output_type: str = "text",
            temperature: float = 0.2,
            top_p: float = 0.9,
            frequency_penalty: float = 1.0,
            presence_penalty: float = 0.0,
            system_prompt: Optional[str] = None,
            search_recency_filter: Optional[str] = None,
            return_citations: bool = False,
            json_mode: bool = False,
            json_schema: Optional[Dict[str, Any]] = None,
            max_input_tokens_default: int = 10_000,
            max_output_tokens_default: int = 10_000,
    ):
        self.base = base
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else max_input_tokens_default
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else max_output_tokens_default
        self.output_type = output_type
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.search_recency_filter = search_recency_filter
        self.return_citations = return_citations
        self.json_mode = json_mode
        self.json_schema = json_schema
        self._max_input_tokens_default = max_input_tokens_default
        self._max_output_tokens_default = max_output_tokens_default

    @property
    def identifier(self) -> str:
        """Generate a unique identifier for this configuration for caching purposes"""
        prompt_str = str(self.prompt)
        truncated_prompt = prompt_str[:30] + "..." if len(prompt_str) > 30 else prompt_str
        base_id = f"{self.base}_{self.model}_{truncated_prompt}"

        hash_input = f"{self.base}_{self.model}_{str(self.prompt)}_{self.max_input_tokens}_{self.max_output_tokens}"
        hash_input += f"_{self.temperature}_{self.top_p}_{self.frequency_penalty}_{self.presence_penalty}"
        hash_input += f"_{self.system_prompt}_{self.search_recency_filter}"
        hash_input += f"_{self.return_citations}_{self.json_mode}"

        if self.json_schema:
            schema_str = json.dumps(self.json_schema, sort_keys=True)
            schema_hash = sha256(schema_str.encode()).hexdigest()[:10]
            hash_input += f"_schema_{schema_hash}"

        _hash = sha256(hash_input.encode()).hexdigest()[:10]
        return f"{base_id}_{_hash}"

    @property
    def safe_config(self) -> Dict[str, Any]:
        """Return a copy of config without sensitive information, suitable for caching"""
        config = {
            "base": self.base,
            "model": self.model,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "output_type": self.output_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "search_recency_filter": self.search_recency_filter,
            "return_citations": self.return_citations,
            "json_mode": self.json_mode        }

        # Add non-empty optional fields
        if self.system_prompt:
            config["system_prompt"] = self.system_prompt

        if self.json_schema:
            config["json_schema"] = self.json_schema

        # Store truncated prompt preview
        if isinstance(self.prompt, str):
            config["prompt_preview"] = (
                self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt
            )
        else:
            config["prompt_preview"] = f"[Conversation with {len(self.prompt)} messages]"

        return config