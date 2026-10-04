from typing import Any

import httpx

from llm_cache_gateway.adapters.base import Adapter
from llm_cache_gateway.config import CONTEXT_MESSAGES

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicShapeAdapter(Adapter):
    """Covers Anthropic's Messages API — auth via x-api-key + anthropic-version
    headers, system prompt as a top-level field, content as text blocks."""

    def __init__(self, base_url: str = "https://api.anthropic.com/v1"):
        self.base_url = base_url.rstrip("/")

    def _message_text(self, content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        return next(block["text"] for block in content if block["type"] == "text")

    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        return self._message_text(request_body["messages"][-1]["content"])

    def extract_cache_key_text(self, request_body: dict[str, Any]) -> str:
        messages = request_body["messages"][-CONTEXT_MESSAGES:]
        return "\n".join(f"{m['role']}: {self._message_text(m['content'])}" for m in messages)

    def build_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        return {
            "id": "msg_cached",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def forward_request(self, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/messages",
            json=request_body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def extract_response_text(self, raw_response: dict[str, Any]) -> str:
        return raw_response["content"][0]["text"]

    def extract_token_usage(self, raw_response: dict[str, Any]) -> int:
        usage = raw_response.get("usage", {})
        return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
