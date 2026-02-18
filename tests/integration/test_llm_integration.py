"""Integration tests for LLMClient with OpenAI provider (requires OPENAI_API_KEY)"""

import pytest
import asyncio
import os
from pydantic import BaseModel
from smartllm import LLMClient, LLMConfig, TextRequest, MessageRequest, Message


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def integration_config():
    """Config for integration tests"""
    return LLMConfig(
        provider="openai",
        default_model="gpt-4o-mini",
        temperature=0,
        max_tokens=100,
    )


@pytest.mark.asyncio
async def test_basic_text_generation(integration_config):
    """Test basic text generation works"""
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Say 'Hello' and nothing else.",
            max_tokens=10,
            temperature=0,
            clear_cache=True
        )
        
        response = await client.generate_text(request)
        
        assert response.text
        assert response.model == "gpt-4o-mini"
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert "hello" in response.text.lower()


@pytest.mark.asyncio
async def test_conversation_messages(integration_config):
    """Test multi-turn conversation"""
    async with LLMClient(integration_config) as client:
        messages = [
            Message(role="user", content="My name is Alice."),
            Message(role="assistant", content="Nice to meet you, Alice!"),
            Message(role="user", content="What's my name?"),
        ]
        
        request = MessageRequest(messages=messages, temperature=0, clear_cache=True)
        response = await client.send_message(request)
        
        assert "alice" in response.text.lower()


@pytest.mark.asyncio
async def test_streaming_response(integration_config):
    """Test streaming text generation"""
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Count from 1 to 3.",
            max_tokens=50,
            stream=True
        )
        
        chunks = []
        async for chunk in client.generate_text_stream(request):
            chunks.append(chunk.text)
        
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert full_text


@pytest.mark.asyncio
async def test_structured_output(integration_config):
    """Test structured output with Pydantic model"""
    
    class Person(BaseModel):
        """A person with name and age"""
        name: str
        age: int
    
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Return a person named John who is 30 years old. Respond in JSON format.",
            response_format=Person,
            temperature=0,
            clear_cache=True
        )
        
        response = await client.generate_text(request)
        
        assert response.structured_data is not None
        assert isinstance(response.structured_data, Person)
        assert response.structured_data.name.lower() == "john"
        assert response.structured_data.age == 30


@pytest.mark.asyncio
async def test_caching_works(integration_config):
    """Test that caching actually works"""
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Say 'cached test' and nothing else.",
            temperature=0,
            use_cache=True
        )
        
        # First call - should hit API
        response1 = await client.generate_text(request)
        
        # Second call - should hit cache
        response2 = await client.generate_text(request)
        
        assert response1.text == response2.text
        assert response1.input_tokens == response2.input_tokens


@pytest.mark.asyncio
async def test_concurrent_requests(integration_config):
    """Test multiple concurrent requests"""
    async with LLMClient(integration_config) as client:
        requests = [
            TextRequest(
                prompt=f"Say 'test {i}' and nothing else.",
                max_tokens=10,
                temperature=1.0,
                use_cache=False
            )
            for i in range(3)
        ]
        
        responses = await asyncio.gather(*[
            client.generate_text(req) for req in requests
        ])
        
        assert len(responses) == 3
        assert all(r.text for r in responses)
