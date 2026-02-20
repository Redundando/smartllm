"""Integration tests for LLMClient (requires API credentials)"""

import pytest
import asyncio
import os
from pydantic import BaseModel
from smartllm import LLMClient, LLMConfig, TextRequest, MessageRequest, Message


@pytest.fixture
def integration_config(test_model, test_provider):
    """Config for integration tests"""
    return LLMConfig(
        provider=test_provider,
        default_model=test_model,
        temperature=0,
        max_tokens=100,
    )


@pytest.mark.asyncio
async def test_basic_text_generation(integration_config, test_model, test_api_type):
    """Test basic text generation works"""
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Say 'Hello' and nothing else.",
            max_tokens=50,
            temperature=0,
            clear_cache=True,
            api_type=test_api_type,
        )

        response = await client.generate_text(request)

        assert response.text
        assert response.model == test_model
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert "hello" in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.no_responses_api
async def test_conversation_messages(integration_config, test_api_type):
    """Test multi-turn conversation"""
    async with LLMClient(integration_config) as client:
        messages = [
            Message(role="user", content="My name is Alice."),
            Message(role="assistant", content="Nice to meet you, Alice!"),
            Message(role="user", content="What's my name?"),
        ]

        request = MessageRequest(messages=messages, temperature=0, clear_cache=True, api_type=test_api_type)
        response = await client.send_message(request)

        assert "alice" in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.no_responses_api
async def test_streaming_response(integration_config, test_api_type):
    """Test streaming text generation"""
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Count from 1 to 3.",
            max_tokens=50,
            stream=True,
            api_type=test_api_type,
        )

        chunks = []
        async for chunk in client.generate_text_stream(request):
            chunks.append(chunk.text)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert full_text


@pytest.mark.asyncio
async def test_structured_output(integration_config, test_api_type):
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
            clear_cache=True,
            api_type=test_api_type,
        )

        response = await client.generate_text(request)

        assert response.structured_data is not None
        assert isinstance(response.structured_data, Person)
        assert response.structured_data.name.lower() == "john"
        assert response.structured_data.age == 30


@pytest.mark.asyncio
async def test_caching_works(integration_config, test_api_type):
    """Test that caching actually works"""
    async with LLMClient(integration_config) as client:
        request = TextRequest(
            prompt="Say 'cached test' and nothing else.",
            temperature=0,
            use_cache=True,
            api_type=test_api_type,
        )

        response1 = await client.generate_text(request)
        response2 = await client.generate_text(request)

        assert response1.text == response2.text
        assert response1.input_tokens == response2.input_tokens


@pytest.mark.asyncio
async def test_concurrent_requests(integration_config, test_api_type):
    """Test multiple concurrent requests"""
    async with LLMClient(integration_config) as client:
        requests = [
            TextRequest(
                prompt=f"Say 'test {i}' and nothing else.",
                max_tokens=50,
                temperature=1.0,
                use_cache=False,
                api_type=test_api_type,
            )
            for i in range(3)
        ]

        responses = await asyncio.gather(*[
            client.generate_text(req) for req in requests
        ])

        assert len(responses) == 3
        assert all(r.text for r in responses)
