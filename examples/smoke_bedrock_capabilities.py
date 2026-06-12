"""Smoke tests for the Bedrock capability-aware refactor.

Runs a small set of real Bedrock calls against the user's default AWS profile
(eu-north-1, Anthropic inference profiles) to verify each public API path works
end-to-end on both "old API" Claude models (Sonnet 4.6) and "new API" Claude
models (Opus 4.7+).

These are not pytest tests — they're a runnable script that prints a tabular
summary so we can eyeball that everything wires up. Run with:

    .venv\\Scripts\\python.exe examples\\smoke_bedrock_capabilities.py

Set ENV var SMOKE_QUICK=1 to skip thinking-heavy tests.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make the package importable when run as `python examples/smoke_bedrock_capabilities.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from smartllm.bedrock import BedrockLLMClient, BedrockConfig
from smartllm.bedrock.capabilities import get_model_capabilities, supports_thinking
from smartllm.models import TextRequest, MessageRequest, Message


SONNET = "eu.anthropic.claude-sonnet-4-6"
OPUS = "eu.anthropic.claude-opus-4-7"
QUICK = os.getenv("SMOKE_QUICK") == "1"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = ""):
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}  {detail}")
    results.append((name, ok, detail))


async def smoke_capability_inspection():
    print("\n=== Capability inspection ===")
    for model in [SONNET, OPUS, "us.anthropic.claude-3-5-sonnet-20241022-v2:0"]:
        caps = get_model_capabilities(model)
        record(
            f"caps({model[:40]})",
            True,
            f"family={caps.family} temp={caps.accepts_temperature} thinking={caps.thinking_mode}",
        )
    record(
        "supports_thinking(opus-4-7)",
        supports_thinking(OPUS) is True,
        "expected True",
    )
    record(
        "supports_thinking(claude-3-5)",
        supports_thinking("us.anthropic.claude-3-5-sonnet-20241022-v2:0") is False,
        "expected False",
    )


async def smoke_generate_text(client: BedrockLLMClient):
    print("\n=== generate_text (standard) ===")
    for model in [SONNET, OPUS]:
        try:
            req = TextRequest(
                prompt="Reply with the single word: pong.",
                model=model,
                max_tokens=64,
                temperature=0.5,  # accepted on Sonnet 4.6, dropped (with warning) on Opus 4.7
                clear_cache=True,
            )
            resp = await client.generate_text(req)
            ok = "pong" in resp.text.lower()
            record(f"generate_text on {model[:30]}", ok, f"text={resp.text!r}")
        except Exception as e:
            record(f"generate_text on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


async def smoke_generate_text_with_thinking(client: BedrockLLMClient):
    if QUICK:
        return
    print("\n=== generate_text with thinking (manual_budget vs adaptive_effort) ===")
    for model in [SONNET, OPUS]:
        try:
            req = TextRequest(
                prompt="What is 17 * 23? Answer with just the number.",
                model=model,
                max_tokens=4096,
                reasoning_effort="low",
                clear_cache=True,
            )
            t0 = time.monotonic()
            resp = await client.generate_text(req)
            elapsed = round(time.monotonic() - t0, 1)
            answer_ok = "391" in resp.text
            record(
                f"thinking on {model[:30]}",
                answer_ok,
                f"text={resp.text.strip()[:30]!r} reasoning_tokens={resp.reasoning_tokens} elapsed={elapsed}s",
            )
        except Exception as e:
            record(f"thinking on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


class Greeting(BaseModel):
    """A short greeting object."""
    word: str = Field(..., description="The greeting word")
    enthusiasm: int = Field(..., description="Enthusiasm level 1-10")


async def smoke_structured_output(client: BedrockLLMClient):
    print("\n=== generate_text with structured output ===")
    for model in [SONNET, OPUS]:
        try:
            req = TextRequest(
                prompt="Return a Greeting with word='hello' and enthusiasm=7.",
                model=model,
                max_tokens=512,
                response_format=Greeting,
                clear_cache=True,
            )
            resp = await client.generate_text(req)
            ok = (
                isinstance(resp.structured_data, Greeting)
                and resp.structured_data.word.lower() == "hello"
                and resp.structured_data.enthusiasm == 7
            )
            record(f"structured on {model[:30]}", ok, f"data={resp.structured_data}")
        except Exception as e:
            record(f"structured on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


async def smoke_thinking_plus_structured(client: BedrockLLMClient):
    if QUICK:
        return
    print("\n=== generate_text with thinking + structured output (two-pass) ===")
    for model in [SONNET, OPUS]:
        try:
            req = TextRequest(
                prompt=(
                    "Compute 17*23 and return the result as a Greeting where "
                    "word is the answer (as text) and enthusiasm is 5."
                ),
                model=model,
                max_tokens=4096,
                reasoning_effort="low",
                response_format=Greeting,
                clear_cache=True,
            )
            resp = await client.generate_text(req)
            ok = isinstance(resp.structured_data, Greeting)
            record(
                f"thinking+structured on {model[:30]}",
                ok,
                f"data={resp.structured_data} reasoning_tokens={resp.reasoning_tokens}",
            )
        except Exception as e:
            record(f"thinking+structured on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


async def smoke_streaming_assembly(client: BedrockLLMClient):
    if QUICK:
        return
    print("\n=== generate_text_streamed (with progress events) ===")
    for model in [SONNET, OPUS]:
        events: list[dict] = []

        async def on_progress(ev: dict):
            events.append(ev)

        try:
            req = TextRequest(
                prompt="Count from 1 to 10, one number per line.",
                model=model,
                max_tokens=512,
                clear_cache=True,
                on_progress=on_progress,
            )
            resp = await client.generate_text_streamed(req)
            ev_types = {e.get("event") for e in events}
            ok = bool(resp.text) and "llm_started" in ev_types and "llm_done" in ev_types
            record(
                f"streamed on {model[:30]}",
                ok,
                f"events={sorted(ev_types)} chars={len(resp.text)}",
            )
        except Exception as e:
            record(f"streamed on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


async def smoke_streaming_with_thinking(client: BedrockLLMClient):
    if QUICK:
        return
    print("\n=== generate_text_streamed with thinking ===")
    for model in [SONNET, OPUS]:
        try:
            req = TextRequest(
                prompt="What is 12 * 13? Answer with just the number.",
                model=model,
                max_tokens=4096,
                reasoning_effort="low",
                clear_cache=True,
            )
            resp = await client.generate_text_streamed(req)
            ok = "156" in resp.text
            record(
                f"streamed+thinking on {model[:30]}",
                ok,
                f"text={resp.text.strip()[:40]!r} reasoning_tokens={resp.reasoning_tokens}",
            )
        except Exception as e:
            record(f"streamed+thinking on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


async def smoke_multi_turn_with_thinking(client: BedrockLLMClient):
    if QUICK:
        return
    print("\n=== send_message with thinking (new MessageRequest field) ===")
    for model in [SONNET, OPUS]:
        try:
            messages = [
                Message(role="user", content="Pick a number between 1 and 100."),
                Message(role="assistant", content="42."),
                Message(role="user", content="Now multiply it by 3 and just give me the number."),
            ]
            req = MessageRequest(
                messages=messages,
                model=model,
                max_tokens=2048,
                reasoning_effort="low",
                clear_cache=True,
            )
            resp = await client.send_message(req)
            ok = "126" in resp.text
            record(
                f"send_message+thinking on {model[:30]}",
                ok,
                f"text={resp.text.strip()[:40]!r}",
            )
        except Exception as e:
            record(f"send_message+thinking on {model[:30]}", False, f"exception: {type(e).__name__}: {e}")


async def smoke_unsupported_thinking_warning(client: BedrockLLMClient):
    """Verify the body builder drops thinking params + warns on non-thinking models.

    We don't hit the API for this — the eu-north-1 default profile doesn't
    publish claude-3.x inference profiles. Instead we exercise the body builder
    directly with a known-non-thinking model ID and assert the body shape.
    """
    print("\n=== thinking on a model that doesn't support it (body-shape only) ===")
    legacy = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    body = client._build_claude_body(
        model=legacy,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        reasoning_effort="low",
    )
    ok = "thinking" not in body and "output_config" not in body
    record(
        "legacy_model_drops_thinking",
        ok,
        f"keys={sorted(body.keys())}",
    )


async def main():
    config = BedrockConfig(aws_region="eu-north-1")
    config.validate()
    print(f"Using region={config.aws_region}, default_model={config.default_model}")
    print(f"Mode: {'QUICK' if QUICK else 'FULL'}")

    await smoke_capability_inspection()

    async with BedrockLLMClient(config=config) as client:
        await smoke_generate_text(client)
        await smoke_structured_output(client)
        await smoke_unsupported_thinking_warning(client)
        await smoke_generate_text_with_thinking(client)
        await smoke_thinking_plus_structured(client)
        await smoke_streaming_assembly(client)
        await smoke_streaming_with_thinking(client)
        await smoke_multi_turn_with_thinking(client)

    print("\n=== Summary ===")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} checks passed")
    for name, ok, detail in results:
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {name}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
