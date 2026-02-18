"""Utilities for converting Pydantic models to LLM tool schemas"""

from typing import Type, Dict, Any
from pydantic import BaseModel


def pydantic_to_tool_schema(model: Type[BaseModel], tool_name: str = None) -> Dict[str, Any]:
    """Convert a Pydantic model to tool schema format (Claude/OpenAI compatible)
    
    Args:
        model: Pydantic BaseModel class
        tool_name: Optional custom tool name (defaults to model name)
        
    Returns:
        Tool schema dict for LLM API
    """
    schema = model.model_json_schema()
    name = tool_name or f"return_{model.__name__.lower()}"
    description = model.__doc__ or f"Returns structured {model.__name__} data"
    
    return {
        "name": name,
        "description": description.strip(),
        "input_schema": {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    }
