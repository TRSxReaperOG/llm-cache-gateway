from typing import Any

import httpx

from llm_cache_gateway.adapters.base import Adapter


class OpenAIShapeAdapter(Adapter):
    """Covers any provider exposing an OpenAI-compatible chat completions API
    (OpenAI, Groq, Mistral, Together) — they share the same request/response shape."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        return request_body["messages"][-1]["content"]

    def build_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        return {
            "id": "chatcmpl-cached",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def forward_request(self, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=request_body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
