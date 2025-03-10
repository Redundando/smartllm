from typing import Union, Optional, Dict, List, Any
from logorator import Logger


class LLMProvider:
    """Base class for LLM providers with template methods for common operations"""

    def create_client(self, api_key: str, base_url: Optional[str] = None) -> Any:
        """Create a client for this provider"""
        raise NotImplementedError("Subclasses must implement create_client")

    @Logger()
    def generate(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
    ) -> Any:
        """Template method for generation with pre/post processing"""
        try:
            Logger.note(f"Sending request to {self._get_provider_name()} API with model: {model}")
            response = self._execute_api_request(client, params)
            Logger.note(f"Received response from {self._get_provider_name()} API")
            return response
        except Exception as e:
            Logger.note(f"Error in {self._get_provider_name()} API request: {str(e)}")
            raise

    def _execute_api_request(self, client: Any, params: Dict[str, Any]) -> Any:
        """Provider-specific API request implementation"""
        raise NotImplementedError("Subclasses must implement _execute_api_request")

    def _get_provider_name(self) -> str:
        """Get the provider name for logging"""
        return self.__class__.__name__.replace('Provider', '')

    def prepare_messages(
            self,
            prompt: Union[str, List[str]],
            system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Convert prompt to messages format"""
        messages = []

        # Handle system prompt if supported by this provider
        if system_prompt and self._supports_system_prompt():
            messages.append({"role": "system", "content": system_prompt})

        # Handle single string prompt
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        # Handle conversation history
        else:
            for i, msg in enumerate(prompt):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})

        return messages

    def _supports_system_prompt(self) -> bool:
        """Whether this provider supports system prompts in message list"""
        return True

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
        """Prepare parameters for API request"""
        # Base parameters common to most providers
        params = {
                "model"      : model,
                "messages"   : messages,
                "temperature": temperature,
                "top_p"      : top_p
        }

        # Provider-specific parameter names
        if max_tokens:
            params[self._get_max_tokens_param_name()] = max_tokens

        # Add frequency and presence penalties if supported
        if self._supports_penalties():
            params["frequency_penalty"] = frequency_penalty
            params["presence_penalty"] = presence_penalty

        # Add search recency filter if supported
        if search_recency_filter and self._supports_search_filter():
            params["search_recency_filter"] = search_recency_filter

        # Add JSON mode configuration if supported and requested
        if json_mode:
            self._configure_json_mode(params, json_schema)

        # Add system prompt if needed outside of messages
        if system_prompt and self._needs_separate_system_prompt():
            params["system"] = system_prompt

        return params

    def _get_max_tokens_param_name(self) -> str:
        """Get the parameter name for max tokens"""
        return "max_tokens"

    def _supports_penalties(self) -> bool:
        """Whether this provider supports frequency and presence penalties"""
        return True

    def _supports_search_filter(self) -> bool:
        """Whether this provider supports search recency filtering"""
        return False

    def _needs_separate_system_prompt(self) -> bool:
        """Whether this provider needs system prompt outside of messages"""
        return False

    def _configure_json_mode(self, params: Dict[str, Any], json_schema: Optional[Dict[str, Any]]) -> None:
        """Configure JSON mode in parameters"""
        params["response_format"] = {"type": "json_object"}

    def format_response(
            self,
            response: Any,
            return_citations: bool
    ) -> Dict[str, Any]:
        """Format raw API response into standardized structure"""
        content = self.extract_content(response)

        result = {
                "content"     : content,
                "model"       : self._extract_model_info(response),
                "id"          : self._extract_response_id(response),
                "usage"       : self._extract_usage_info(response),
                "raw_response": response
        }

        # Add citations if requested and available
        if return_citations and self._has_citations(response):
            result["citations"] = self._extract_citations(response)

        return result

    def _extract_model_info(self, response: Any) -> str:
        """Extract model information from response"""
        raise NotImplementedError("Subclasses must implement _extract_model_info")

    def _extract_response_id(self, response: Any) -> str:
        """Extract response ID from response"""
        raise NotImplementedError("Subclasses must implement _extract_response_id")

    def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        """Extract token usage information from response"""
        raise NotImplementedError("Subclasses must implement _extract_usage_info")

    def _has_citations(self, response: Any) -> bool:
        """Check if response has citations"""
        return hasattr(response, 'citations')

    def _extract_citations(self, response: Any) -> List[str]:
        """Extract citations from response"""
        return getattr(response, 'citations', [])

    def format_json_response(
            self,
            response: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract JSON content from response if available"""
        return None

    def extract_content(self, raw_response: Any) -> str:
        """Extract the main text content from response"""
        raise NotImplementedError("Subclasses must implement extract_content")

    def create_serializable_response(
            self,
            raw_response: Any,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        """Create a response object that can be serialized to JSON"""
        content = self.extract_content(raw_response)

        serializable = {
                "content"  : content,
                "model"    : self._extract_model_info(raw_response),
                "id"       : self._extract_response_id(raw_response),
                "usage"    : self._extract_usage_info(raw_response),
                "citations": self._extract_citations(raw_response) if self._has_citations(raw_response) else []
        }

        # Add JSON content if in JSON mode
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
        """Count tokens for the given messages and model"""
        raise NotImplementedError("Subclasses must implement count_tokens")

    def list_models(
            self,
            client: Any,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List available models for this provider"""
        raise NotImplementedError("Subclasses must implement list_models")