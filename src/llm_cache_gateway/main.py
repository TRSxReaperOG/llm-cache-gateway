import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_cache_gateway.adapters.anthropic_shape import AnthropicShapeAdapter
from llm_cache_gateway.adapters.base import Adapter
from llm_cache_gateway.adapters.cohere_shape import CohereShapeAdapter
from llm_cache_gateway.adapters.gemini_shape import GeminiShapeAdapter
from llm_cache_gateway.adapters.openai_shape import OpenAIShapeAdapter
from llm_cache_gateway.db import (
    clear_all,
    delete_entry,
    ensure_collection,
    get_stats,
    list_recent,
    record_hit,
    search,
    store,
)
from llm_cache_gateway.embeddings import embed_text
from llm_cache_gateway.pricing import estimate_cost

logger = logging.getLogger("llm_cache_gateway")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

app = FastAPI()

ADAPTERS = {
    "openai": OpenAIShapeAdapter(base_url="https://api.openai.com/v1"),
    "groq": OpenAIShapeAdapter(base_url="https://api.groq.com/openai/v1"),
    "mistral": OpenAIShapeAdapter(base_url="https://api.mistral.ai/v1"),
    "together": OpenAIShapeAdapter(base_url="https://api.together.xyz/v1"),
    "anthropic": AnthropicShapeAdapter(),
    "gemini": GeminiShapeAdapter(),
    "cohere": CohereShapeAdapter(),
}
# Picked from eval/THRESHOLD_DECISION.md: lowest threshold with zero false
# positives on the labeled paraphrase/confusable-pair eval set.
SIMILARITY_THRESHOLD = 0.93
STREAM_CHUNK_DELAY_SECONDS = 0.02

ensure_collection()


def log_request(
    request_id: str, provider: str, hit: bool, latency_ms: float, tokens: int, cost_estimate_usd: float
) -> None:
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "provider": provider,
                "hit": hit,
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens,
                "cost_estimate_usd": round(cost_estimate_usd, 6),
            }
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/stats")
def admin_stats() -> dict[str, Any]:
    return get_stats()


@app.get("/admin/entries")
def admin_entries(limit: int = 20) -> list[dict[str, Any]]:
    return list_recent(limit=limit)


@app.delete("/admin/cache")
def admin_clear_cache() -> dict[str, str]:
    clear_all()
    return {"status": "cleared"}


@app.delete("/admin/cache/{entry_id}")
def admin_delete_entry(entry_id: str) -> dict[str, str]:
    delete_entry(entry_id)
    return {"status": "deleted", "id": entry_id}


async def _simulated_hit_stream(adapter: Adapter, cached_text: str) -> AsyncIterator[str]:
    words = cached_text.split(" ")
    for i, word in enumerate(words):
        fragment = word + (" " if i < len(words) - 1 else "")
        yield f"{adapter.format_stream_chunk(fragment)}\n\n"
        await asyncio.sleep(STREAM_CHUNK_DELAY_SECONDS)
    yield "data: [DONE]\n\n"


async def _passthrough_miss_stream(
    adapter: Adapter,
    body: dict[str, Any],
    api_key: str,
    vector: list[float],
    cache_key_text: str,
    provider: str,
    request_id: str,
    start: float,
) -> AsyncIterator[str]:
    full_text = []
    for line in adapter.forward_request_stream(body, api_key):
        yield f"{line}\n\n"
        text = adapter.extract_stream_text(line)
        if text:
            full_text.append(text)
    yield "data: [DONE]\n\n"

    response_text = "".join(full_text)
    store(vector, cache_key_text, response_text, provider=provider, tokens=0)
    log_request(request_id, provider, False, (time.monotonic() - start) * 1000, 0, 0.0)


@app.post("/v1/proxy/{provider}/chat/completions")
async def chat_completions(provider: str, request: Request) -> Any:
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider!r}")

    body = await request.json()
    api_key = request.headers["authorization"].removeprefix("Bearer ").strip()
    request_id = str(uuid.uuid4())
    start = time.monotonic()
    wants_stream = bool(body.get("stream"))

    if wants_stream and not adapter.supports_streaming:
        raise HTTPException(status_code=501, detail=f"Streaming not supported for provider {provider!r}")

    prompt = adapter.extract_prompt(body)
    cache_key_text = adapter.extract_cache_key_text(body)
    vector = embed_text(cache_key_text)

    matches = search(vector)
    if matches and matches[0].score >= SIMILARITY_THRESHOLD:
        match = matches[0]
        record_hit(str(match.id))
        cached_response_text = (match.payload or {})["response"]
        log_request(request_id, provider, True, (time.monotonic() - start) * 1000, 0, 0.0)

        if wants_stream:
            stream = _simulated_hit_stream(adapter, cached_response_text)
            return StreamingResponse(stream, media_type="text/event-stream")
        return adapter.build_response(prompt, cached_response_text)

    if wants_stream:
        return StreamingResponse(
            _passthrough_miss_stream(adapter, body, api_key, vector, cache_key_text, provider, request_id, start),
            media_type="text/event-stream",
        )

    response = adapter.forward_request(body, api_key)
    response_text = adapter.extract_response_text(response)
    tokens = adapter.extract_token_usage(response)
    store(vector, cache_key_text, response_text, provider=provider, tokens=tokens)
    log_request(request_id, provider, False, (time.monotonic() - start) * 1000, tokens, estimate_cost(provider, tokens))

    return response
