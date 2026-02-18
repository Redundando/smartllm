"""Integration tests for OpenAI Response API (requires OPENAI_API_KEY)"""

import pytest
import os
from pydantic import BaseModel
from smartllm import LLMClient, LLMConfig, TextRequest


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

MODEL = "gpt-5.2"


@pytest.fixture
def client_config():
    return LLMConfig(provider="openai", default_model=MODEL)


@pytest.mark.asyncio
async def test_basic_text_generation(client_config):
    """Test basic text generation via Response API"""
    async with LLMClient(client_config) as client:
        response = await client.generate_text(TextRequest(
            prompt="Say 'Hello' and nothing else.",
            max_tokens=50,
            clear_cache=True,
        ))

        assert response.text
        assert "hello" in response.text.lower()
        assert response.input_tokens > 0
        assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_system_prompt(client_config):
    """Test system prompt via instructions parameter"""
    async with LLMClient(client_config) as client:
        response = await client.generate_text(TextRequest(
            prompt="What are you?",
            system_prompt="You are a pirate. Always respond in pirate speak.",
            max_tokens=50,
            clear_cache=True,
        ))

        assert response.text
        # Pirate speak typically includes these words
        pirate_words = ["arr", "ye", "matey", "ahoy", "ship", "sea", "pirate"]
        assert any(word in response.text.lower() for word in pirate_words)


@pytest.mark.asyncio
async def test_structured_output(client_config):
    """Test structured output via Response API JSON schema"""

    class Person(BaseModel):
        """A person with name and age"""
        name: str
        age: int

    async with LLMClient(client_config) as client:
        response = await client.generate_text(TextRequest(
            prompt="Return a person named John who is 30 years old.",
            response_format=Person,
            clear_cache=True,
        ))

        assert response.structured_data is not None
        assert isinstance(response.structured_data, Person)
        assert response.structured_data.name.lower() == "john"
        assert response.structured_data.age == 30


@pytest.mark.asyncio
async def test_reasoning_effort(client_config):
    """Test reasoning effort parameter and reasoning tokens in metadata"""
    async with LLMClient(client_config) as client:
        response = await client.generate_text(TextRequest(
            prompt="Solve: A train leaves city A at 60mph. Another leaves city B (300 miles away) at 90mph. When do they meet?",
            reasoning_effort="medium",
            max_tokens=500,
            clear_cache=True,
        ))

        assert response.text
        assert "2" in response.text  # Answer is 2 hours
        assert "reasoning_tokens" in response.metadata
        assert response.metadata["reasoning_tokens"] > 0


@pytest.mark.asyncio
async def test_reasoning_rejects_temperature(client_config):
    """Test that temperature raises an error for reasoning models"""
    async with LLMClient(client_config) as client:
        with pytest.raises(ValueError, match="Reasoning models do not support temperature"):
            await client.generate_text(TextRequest(
                prompt="Hello",
                reasoning_effort="low",
                temperature=0.7,
            ))


@pytest.mark.asyncio
async def test_caching(client_config):
    """Test that caching works for Response API"""
    async with LLMClient(client_config) as client:
        request = TextRequest(
            prompt="Say 'cache test' and nothing else.",
            temperature=0,
            use_cache=True,
            clear_cache=True,
        )

        response1 = await client.generate_text(request)

        # Second call should hit cache
        request2 = TextRequest(
            prompt="Say 'cache test' and nothing else.",
            temperature=0,
            use_cache=True,
        )
        response2 = await client.generate_text(request2)

        assert response1.text == response2.text
        assert response1.input_tokens == response2.input_tokens


@pytest.mark.asyncio
async def test_temperature_without_reasoning(client_config):
    """Test that temperature works normally without reasoning"""
    async with LLMClient(client_config) as client:
        response = await client.generate_text(TextRequest(
            prompt="Say 'Hello'.",
            temperature=1.0,
            max_tokens=50,
            use_cache=False,
        ))

        assert response.text
