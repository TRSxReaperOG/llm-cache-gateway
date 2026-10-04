import json
from collections.abc import Iterator
from typing import Any

import httpx

from llm_cache_gateway.adapters.base import Adapter
from llm_cache_gateway.config import CONTEXT_MESSAGES


class OpenAIShapeAdapter(Adapter):
    """Covers any provider exposing an OpenAI-compatible chat completions API
    (OpenAI, Groq, Mistral, Together) — they share the same request/response shape."""

    supports_streaming = True

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        return request_body["messages"][-1]["content"]

    def extract_cache_key_text(self, request_body: dict[str, Any]) -> str:
        messages = request_body["messages"][-CONTEXT_MESSAGES:]
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

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

    def extract_response_text(self, raw_response: dict[str, Any]) -> str:
        return raw_response["choices"][0]["message"]["content"]

    def extract_token_usage(self, raw_response: dict[str, Any]) -> int:
        return raw_response.get("usage", {}).get("total_tokens", 0)

    def forward_request_stream(self, request_body: dict[str, Any], api_key: str) -> Iterator[str]:
        body = {**request_body, "stream": True}
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield line

    def extract_stream_text(self, line: str) -> str | None:
        data = line[len("data: ") :]
        if data.strip() == "[DONE]":
            return None
        delta = json.loads(data)["choices"][0].get("delta", {})
        return delta.get("content")

    def format_stream_chunk(self, text_fragment: str) -> str:
        payload = {"choices": [{"index": 0, "delta": {"content": text_fragment}, "finish_reason": None}]}
        return f"data: {json.dumps(payload)}"
