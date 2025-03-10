from typing import Union, Optional, Dict, List, Any
from logorator import Logger


class LLMProvider:
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
        """Generate a response from the provider"""
        Logger.note(f"Sending request to {self.__class__.__name__} API with model: {model}")
        response = self._execute_request(client, params)
        Logger.note(f"Received response from {self.__class__.__name__} API")
        return response

    def _execute_request(self, client: Any, params: Dict[str, Any]) -> Any:
        """Execute the actual API request (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _execute_request")

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

        # Add max tokens parameter
        if max_tokens:
            params["max_tokens"] = max_tokens

        # Add frequency and presence penalties
        params["frequency_penalty"] = frequency_penalty
        params["presence_penalty"] = presence_penalty

        # Add search recency filter if provided
        if search_recency_filter and search_recency_filter in ["month", "week", "day", "hour"]:
            params["search_recency_filter"] = search_recency_filter

        # Add JSON mode configuration if requested
        if json_mode:
            if json_schema:
                self._configure_json_mode_with_schema(params, json_schema)
            else:
                params["response_format"] = {"type": "json_object"}

        # Add system prompt if needed outside of messages
        if system_prompt and not self._supports_system_prompt():
            params["system"] = system_prompt

        return params

    def _configure_json_mode_with_schema(self, params: Dict[str, Any], json_schema: Dict[str, Any]) -> None:
        """Configure JSON mode with schema in parameters (to be overridden by subclasses)"""
        params["response_format"] = {"type": "json_object"}

    def extract_content(self, raw_response: Any) -> str:
        """Extract the main text content from response (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement extract_content")

    def extract_json_content(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        """Extract JSON content from response if available (to be implemented by subclasses)"""
        return None

    def create_response(
            self,
            raw_response: Any,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        """Create a response object from the raw API response"""
        content = self.extract_content(raw_response)

        # Extract basic response metadata
        model = self._extract_model_info(raw_response)
        response_id = self._extract_response_id(raw_response)
        usage = self._extract_usage_info(raw_response)

        # Create the response object
        response = {
                "content"  : content,
                "model"    : model,
                "id"       : response_id,
                "usage"    : usage,
                "citations": self._extract_citations(raw_response)
        }

        # Add JSON content if in JSON mode
        if json_mode:
            json_content = self.extract_json_content(raw_response)
            if json_content:
                response["json_content"] = json_content

        return response

    def _extract_model_info(self, response: Any) -> str:
        """Extract model information from response (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _extract_model_info")

    def _extract_response_id(self, response: Any) -> str:
        """Extract response ID from response (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _extract_response_id")

    def _extract_usage_info(self, response: Any) -> Dict[str, int]:
        """Extract token usage information from response (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _extract_usage_info")

    def _extract_citations(self, response: Any) -> List[str]:
        """Extract citations from response"""
        return getattr(response, 'citations', [])