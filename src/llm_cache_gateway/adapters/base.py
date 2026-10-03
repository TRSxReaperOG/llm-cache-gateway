from abc import ABC, abstractmethod
from typing import Any


class Adapter(ABC):
    @abstractmethod
    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        """Pull the plain-text prompt out of a provider-shaped request body."""

    @abstractmethod
    def build_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        """Shape a cache-hit response to look like this provider's real API response."""

    @abstractmethod
    def forward_request(self, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
        """Send the request upstream to the real provider, return its raw response body."""
