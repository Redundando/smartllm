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

    # Resolve all $ref pointers inline — Anthropic doesn't reliably handle $defs
    defs = schema.get("$defs", {})
    resolved = _resolve_refs(schema, defs)
    resolved.pop("$defs", None)

    return {
        "name": name,
        "description": description.strip(),
        "input_schema": {
            "type": resolved.get("type", "object"),
            "properties": resolved.get("properties", {}),
            "required": resolved.get("required", []),
        },
    }


def _resolve_refs(obj: Any, defs: Dict[str, Any]) -> Any:
    """Recursively resolve all $ref pointers by inlining the referenced schema."""
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_name = obj["$ref"].split("/")[-1]
            if ref_name in defs:
                return _resolve_refs(defs[ref_name], defs)
            return obj
        return {k: _resolve_refs(v, defs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_refs(item, defs) for item in obj]
    return obj
