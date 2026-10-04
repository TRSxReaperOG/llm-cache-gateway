from typing import Any

import httpx

from llm_cache_gateway.adapters.base import Adapter


class CohereShapeAdapter(Adapter):
    """Covers Cohere's v2 Chat API — messages/content shape close to OpenAI's,
    auth via a standard Authorization: Bearer header."""

    def __init__(self, base_url: str = "https://api.cohere.ai/v2"):
        self.base_url = base_url.rstrip("/")

    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        return request_body["messages"][-1]["content"]

    def build_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        return {
            "id": "cached",
            "message": {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
            "finish_reason": "COMPLETE",
            "usage": {"billed_units": {"input_tokens": 0, "output_tokens": 0}},
        }

    def forward_request(self, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/chat",
            json=request_body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def extract_response_text(self, raw_response: dict[str, Any]) -> str:
        return raw_response["message"]["content"][0]["text"]

    def extract_token_usage(self, raw_response: dict[str, Any]) -> int:
        billed_units = raw_response.get("usage", {}).get("billed_units", {})
        return billed_units.get("input_tokens", 0) + billed_units.get("output_tokens", 0)
