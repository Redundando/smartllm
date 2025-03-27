import asyncio

from smartllm import AsyncLLM


async def main():
    json_schema = {
            "type"      : "object",
            "properties": {
                    "content_analysis": {
                            "type" : "array",
                            "items": {
                                    "type"      : "object",
                                    "properties": {
                                            "content_name"   : {"type": "string"},
                                            "coverage_rating": {"type": "integer", "minimum": 0, "maximum": 10},
                                            "analysis_notes" : {"type": "string"}},
                                    "required"  : ["content_name", "coverage_rating"]}}}}
    asy = AsyncLLM(
        base="openai",
        model="gpt-4o",
        prompt="Please analyse the content of Godot.",
        stream=False,
        json_schema=json_schema,
        )
    print(await asy.models())


if __name__ == "__main__":
    asyncio.run(main())
