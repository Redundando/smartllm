from typing import Union, Optional, Dict, List, Any
from logorator import Logger

class LLMProvider:
    def create_client(self, api_key: str, base_url: Optional[str] = None) -> Any:
        pass

    @Logger()
    def generate(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            params: Dict[str, Any],
    ) -> Any:
        pass

    def prepare_messages(
            self,
            prompt: Union[str, List[str]],
            system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        pass

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
            system_prompt: Optional[str] = None,
            stream: bool = False,
    ) -> Dict[str, Any]:
        pass

    def format_response(
            self,
            response: Any,
            return_citations: bool
    ) -> Dict[str, Any]:
        pass

    def format_json_response(
            self,
            response: Any
    ) -> Optional[Dict[str, Any]]:
        return None

    def extract_content(self, raw_response: Any) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def create_serializable_response(
            self,
            raw_response: Any,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    def count_tokens(
            self,
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None
    ) -> int:
        pass

    def list_models(
            self,
            client: Any,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        pass