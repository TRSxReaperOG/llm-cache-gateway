from typing import Any

import httpx

from llm_cache_gateway.adapters.base import Adapter
from llm_cache_gateway.config import CONTEXT_MESSAGES


class GeminiShapeAdapter(Adapter):
    """Covers Google's Gemini generateContent API — contents/parts request
    shape, model in the URL path, auth via x-goog-api-key header."""

    def __init__(self, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.base_url = base_url.rstrip("/")

    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        parts = request_body["contents"][-1]["parts"]
        return "".join(part["text"] for part in parts if "text" in part)

    def extract_cache_key_text(self, request_body: dict[str, Any]) -> str:
        contents = request_body["contents"][-CONTEXT_MESSAGES:]
        lines = []
        for entry in contents:
            text = "".join(part["text"] for part in entry["parts"] if "text" in part)
            lines.append(f"{entry['role']}: {text}")
        return "\n".join(lines)

    def build_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": response_text}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0,
            },
        }

    def forward_request(self, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
        model = request_body["model"]
        contents = {k: v for k, v in request_body.items() if k != "model"}

        response = httpx.post(
            f"{self.base_url}/models/{model}:generateContent",
            json=contents,
            headers={"x-goog-api-key": api_key},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def extract_response_text(self, raw_response: dict[str, Any]) -> str:
        return raw_response["candidates"][0]["content"]["parts"][0]["text"]

    def extract_token_usage(self, raw_response: dict[str, Any]) -> int:
        return raw_response.get("usageMetadata", {}).get("totalTokenCount", 0)
