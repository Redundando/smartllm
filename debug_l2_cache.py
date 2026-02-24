import asyncio
from smartllm import LLMClient, TextRequest

TABLE = "audible-toolkit-llm"
PROMPT = "Reply with just: hello"

async def main():
    async with LLMClient(provider="bedrock", dynamo_table_name=TABLE) as client:
        r1 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0, clear_cache=True))
        print(f"API call  [{r1.cache_source}] key={r1.cache_key}: {r1.text}")

        # Clear local cache to force L2 hit
        client._client.cache.local.clear(r1.cache_key)

        r2 = await client.generate_text(TextRequest(prompt=PROMPT, temperature=0))
        print(f"Cache hit [{r2.cache_source}] key={r2.cache_key}: {r2.text}")
        assert r2.cache_source == "l2", f"Expected l2, got {r2.cache_source}"
        print("OK")

        raw = client._client.cache._dynamo.get(r2.cache_key)
        import json
        print(json.dumps(raw, indent=2, default=str))

asyncio.run(main())
