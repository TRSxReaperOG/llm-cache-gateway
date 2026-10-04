from typing import Any

from fastapi import FastAPI, HTTPException, Request

from llm_cache_gateway.adapters.anthropic_shape import AnthropicShapeAdapter
from llm_cache_gateway.adapters.cohere_shape import CohereShapeAdapter
from llm_cache_gateway.adapters.gemini_shape import GeminiShapeAdapter
from llm_cache_gateway.adapters.openai_shape import OpenAIShapeAdapter
from llm_cache_gateway.db import ensure_collection, search, store
from llm_cache_gateway.embeddings import embed_text

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
SIMILARITY_THRESHOLD = 0.95

ensure_collection()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/proxy/{provider}/chat/completions")
async def chat_completions(provider: str, request: Request) -> dict[str, Any]:
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider!r}")

    body = await request.json()
    api_key = request.headers["authorization"].removeprefix("Bearer ").strip()

    prompt = adapter.extract_prompt(body)
    vector = embed_text(prompt)

    matches = search(vector)
    if matches and matches[0].score >= SIMILARITY_THRESHOLD:
        return adapter.build_response(prompt, matches[0].payload["response"])

    response = adapter.forward_request(body, api_key)
    response_text = adapter.extract_response_text(response)
    tokens = adapter.extract_token_usage(response)
    store(vector, prompt, response_text, provider=provider, tokens=tokens)

    return response
