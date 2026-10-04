from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class Adapter(ABC):
    #: Overridden to True by adapters that implement the streaming methods below.
    supports_streaming: bool = False

    @abstractmethod
    def extract_prompt(self, request_body: dict[str, Any]) -> str:
        """Pull the plain-text prompt out of a provider-shaped request body."""

    @abstractmethod
    def build_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        """Shape a cache-hit response to look like this provider's real API response."""

    @abstractmethod
    def forward_request(self, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
        """Send the request upstream to the real provider, return its raw response body."""

    @abstractmethod
    def extract_response_text(self, raw_response: dict[str, Any]) -> str:
        """Pull the plain-text reply out of a provider-shaped response body."""

    @abstractmethod
    def extract_token_usage(self, raw_response: dict[str, Any]) -> int:
        """Pull the total token count out of a provider-shaped response body."""

    @abstractmethod
    def extract_cache_key_text(self, request_body: dict[str, Any]) -> str:
        """Build the text to embed for cache lookup, folding in recent
        conversation context so multi-turn follow-ups don't collide with
        unrelated conversations that happen to end in the same message."""

    def forward_request_stream(self, request_body: dict[str, Any], api_key: str) -> Iterator[str]:
        """Stream raw SSE 'data: ...' lines from the upstream provider as they arrive."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def extract_stream_text(self, line: str) -> str | None:
        """Pull the incremental text out of one raw SSE line, or None if it carries no text."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def format_stream_chunk(self, text_fragment: str) -> str:
        """Wrap a fragment of cached text as one SSE 'data: ...' line in this provider's shape."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
