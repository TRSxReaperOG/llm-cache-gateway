from typing import Any

from fastapi import FastAPI, Request

from llm_cache_gateway.adapters.openai_shape import OpenAIShapeAdapter
from llm_cache_gateway.db import ensure_collection, search, store
from llm_cache_gateway.embeddings import embed_text

app = FastAPI()

GROQ_ADAPTER = OpenAIShapeAdapter(base_url="https://api.groq.com/openai/v1")
SIMILARITY_THRESHOLD = 0.95

ensure_collection()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.json()
    api_key = request.headers["authorization"].removeprefix("Bearer ").strip()

    prompt = GROQ_ADAPTER.extract_prompt(body)
    vector = embed_text(prompt)

    matches = search(vector)
    if matches and matches[0].score >= SIMILARITY_THRESHOLD:
        return GROQ_ADAPTER.build_response(prompt, matches[0].payload["response"])

    response = GROQ_ADAPTER.forward_request(body, api_key)
    response_text = response["choices"][0]["message"]["content"]
    tokens = response.get("usage", {}).get("total_tokens", 0)
    store(vector, prompt, response_text, provider="groq", tokens=tokens)

    return response
