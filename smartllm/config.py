from typing import Union, Optional, Dict, List, Any
from hashlib import sha256
import json


class Configuration:
    # Default values as class constants
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_P = 0.9
    DEFAULT_FREQUENCY_PENALTY = 1.0
    DEFAULT_PRESENCE_PENALTY = 0.0
    DEFAULT_MAX_TOKENS = 10_000
    DEFAULT_OUTPUT_TYPE = "text"

    def __init__(
            self,
            base: str = "",
            model: str = "",
            api_key: str = "",
            prompt: Union[str, List[str]] = "",
            max_input_tokens: Optional[int] = None,
            max_output_tokens: Optional[int] = None,
            output_type: str = DEFAULT_OUTPUT_TYPE,
            temperature: float = DEFAULT_TEMPERATURE,
            top_p: float = DEFAULT_TOP_P,
            frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
            presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
            system_prompt: Optional[str] = None,
            search_recency_filter: Optional[str] = None,
            return_citations: bool = False,
            json_mode: bool = False,
            json_schema: Optional[Dict[str, Any]] = None,
            **kwargs  # Accept and ignore additional kwargs
    ):
        # Core parameters
        self.base = base
        self.model = model
        self.api_key = api_key
        self.prompt = prompt

        # Apply defaults for optional parameters
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else self.DEFAULT_MAX_TOKENS
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else self.DEFAULT_MAX_TOKENS
        self.output_type = output_type
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # Optional parameters
        self.system_prompt = system_prompt
        self.search_recency_filter = search_recency_filter
        self.return_citations = return_citations
        self.json_mode = json_mode
        self.json_schema = json_schema

        # Ignore any additional kwargs (ttl, clear_cache, etc.)

    @property
    def identifier(self) -> str:
        """Generate a unique identifier for this configuration for caching purposes"""
        prompt_str = str(self.prompt)
        truncated_prompt = prompt_str[:30] + "..." if len(prompt_str) > 30 else prompt_str
        base_id = f"{self.base}_{self.model}_{truncated_prompt}"

        # Create hash input string from all relevant parameters
        hash_input = self._create_hash_input()

        # Generate hash
        _hash = sha256(hash_input.encode()).hexdigest()[:10]
        return f"{base_id}_{_hash}"

    @property
    def readable_identifier(self) -> str:
        return f"{self.base}  ({self.prompt[:60]}){' (JSON)' if self.json_mode else ''}"

    def _create_hash_input(self) -> str:
        """Create a string for hashing that incorporates all relevant parameters"""
        hash_input = f"{self.base}_{self.model}_{str(self.prompt)}_{self.max_input_tokens}_{self.max_output_tokens}"
        hash_input += f"_{self.temperature}_{self.top_p}_{self.frequency_penalty}_{self.presence_penalty}"
        hash_input += f"_{self.system_prompt}_{self.search_recency_filter}"
        hash_input += f"_{self.return_citations}_{self.json_mode}"

        # Add schema hash if present
        if self.json_schema:
            schema_str = json.dumps(self.json_schema, sort_keys=True)
            schema_hash = sha256(schema_str.encode()).hexdigest()[:10]
            hash_input += f"_schema_{schema_hash}"

        return hash_input

    @property
    def safe_config(self) -> Dict[str, Any]:
        """Return a copy of config without sensitive information, suitable for caching"""
        # Basic configuration parameters
        config = {
                "base"                 : self.base,
                "model"                : self.model,
                "max_input_tokens"     : self.max_input_tokens,
                "max_output_tokens"    : self.max_output_tokens,
                "output_type"          : self.output_type,
                "temperature"          : self.temperature,
                "top_p"                : self.top_p,
                "frequency_penalty"    : self.frequency_penalty,
                "presence_penalty"     : self.presence_penalty,
                "search_recency_filter": self.search_recency_filter,
                "return_citations"     : self.return_citations,
                "json_mode"            : self.json_mode
        }

        # Add non-empty optional fields
        if self.system_prompt:
            config["system_prompt"] = self.system_prompt

        if self.json_schema:
            config["json_schema"] = self.json_schema

        # Create prompt preview
        config["prompt_preview"] = self._create_prompt_preview()

        return config

    def _create_prompt_preview(self) -> str:
        """Create a safe preview of the prompt for logging and display"""
        if isinstance(self.prompt, str):
            return self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt
        else:
            return f"[Conversation with {len(self.prompt)} messages]"