"""Unit tests for schema utilities"""

import pytest
from pydantic import BaseModel, Field
from smartllm.utils import pydantic_to_tool_schema


class SimpleModel(BaseModel):
    """A simple test model"""
    name: str
    age: int


class ComplexModel(BaseModel):
    """A complex test model"""
    title: str = Field(description="The title")
    count: int = Field(default=0, description="The count")
    tags: list[str] = Field(default_factory=list)


def test_simple_model_conversion():
    """Test converting simple Pydantic model to tool schema"""
    schema = pydantic_to_tool_schema(SimpleModel)
    
    assert schema["name"] == "return_simplemodel"
    assert "simple test model" in schema["description"].lower()
    assert schema["input_schema"]["type"] == "object"
    assert "name" in schema["input_schema"]["properties"]
    assert "age" in schema["input_schema"]["properties"]
    assert "name" in schema["input_schema"]["required"]
    assert "age" in schema["input_schema"]["required"]


def test_complex_model_conversion():
    """Test converting complex Pydantic model with defaults"""
    schema = pydantic_to_tool_schema(ComplexModel)
    
    assert "title" in schema["input_schema"]["properties"]
    assert "count" in schema["input_schema"]["properties"]
    assert "tags" in schema["input_schema"]["properties"]
    assert "title" in schema["input_schema"]["required"]


def test_custom_tool_name():
    """Test custom tool name"""
    schema = pydantic_to_tool_schema(SimpleModel, tool_name="custom_tool")
    
    assert schema["name"] == "custom_tool"
