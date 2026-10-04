import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

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
LATENCY_SAMPLE_WINDOW = 50

ensure_collection()

DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

# Rolling in-memory latency samples and request feed for the dashboard.
# Presentational only — not persisted, resets on restart, not a substitute
# for the structured logs. REQUEST_LOG covers every request (hit or miss);
# Qdrant only stores one row per unique prompt (misses), so it alone can't
# show a live feed where repeated hits visibly appear.
HIT_LATENCIES_MS: deque[float] = deque(maxlen=LATENCY_SAMPLE_WINDOW)
MISS_LATENCIES_MS: deque[float] = deque(maxlen=LATENCY_SAMPLE_WINDOW)
REQUEST_LOG: deque[dict[str, Any]] = deque(maxlen=30)


def log_request(
    request_id: str, provider: str, hit: bool, latency_ms: float, tokens: int, cost_estimate_usd: float, prompt: str
) -> None:
    (HIT_LATENCIES_MS if hit else MISS_LATENCIES_MS).append(latency_ms)
    REQUEST_LOG.appendleft({"timestamp": time.time(), "provider": provider, "hit": hit, "prompt": prompt})
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
    stats = get_stats()

    miss_count = stats["total_entries"]  # every stored entry is exactly one past miss
    hit_count = stats["total_hits"]
    total_requests = miss_count + hit_count
    hit_rate = hit_count / total_requests if total_requests else 0.0

    spent_usd = sum(estimate_cost(provider, tokens) for provider, tokens in stats["tokens_per_provider"].items())
    avg_cost_per_miss = spent_usd / miss_count if miss_count else 0.0
    # Estimate: what a hit would have cost if it had been a real call instead,
    # using the average observed cost per real call. Same method as
    # benchmark/plot_results.py's cost-saved chart.
    estimated_saved_usd = hit_count * avg_cost_per_miss

    avg_hit_latency_ms = sum(HIT_LATENCIES_MS) / len(HIT_LATENCIES_MS) if HIT_LATENCIES_MS else None
    avg_miss_latency_ms = sum(MISS_LATENCIES_MS) / len(MISS_LATENCIES_MS) if MISS_LATENCIES_MS else None

    return {
        **stats,
        "total_requests": total_requests,
        "hit_rate": hit_rate,
        "estimated_cost_spent_usd": round(spent_usd, 6),
        "estimated_cost_saved_usd": round(estimated_saved_usd, 6),
        "avg_hit_latency_ms": round(avg_hit_latency_ms, 2) if avg_hit_latency_ms is not None else None,
        "avg_miss_latency_ms": round(avg_miss_latency_ms, 2) if avg_miss_latency_ms is not None else None,
        "hit_latency_samples": len(HIT_LATENCIES_MS),
        "miss_latency_samples": len(MISS_LATENCIES_MS),
    }


@app.get("/admin/entries")
def admin_entries(limit: int = 20) -> list[dict[str, Any]]:
    return list_recent(limit=limit)


@app.get("/admin/recent-requests")
def admin_recent_requests() -> list[dict[str, Any]]:
    """Live per-request feed (hits and misses both), for the dashboard —
    distinct from /admin/entries, which only lists stored cache entries."""
    return list(REQUEST_LOG)


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
    log_request(request_id, provider, False, (time.monotonic() - start) * 1000, 0, 0.0, cache_key_text)


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
        log_request(request_id, provider, True, (time.monotonic() - start) * 1000, 0, 0.0, cache_key_text)

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
    cost_estimate = estimate_cost(provider, tokens)
    log_request(request_id, provider, False, (time.monotonic() - start) * 1000, tokens, cost_estimate, cache_key_text)

    return response
