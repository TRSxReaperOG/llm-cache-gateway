from unittest.mock import MagicMock, patch

import pytest

from llm_cache_gateway.adapters.anthropic_shape import AnthropicShapeAdapter
from llm_cache_gateway.adapters.cohere_shape import CohereShapeAdapter
from llm_cache_gateway.adapters.gemini_shape import GeminiShapeAdapter
from llm_cache_gateway.adapters.openai_shape import OpenAIShapeAdapter


def test_openai_shape_extract_prompt():
    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    body = {"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "bye"}]}
    assert adapter.extract_prompt(body) == "bye"


def test_openai_shape_cache_key_text_folds_recent_context_only():
    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    body = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
    }
    key = adapter.extract_cache_key_text(body)  # CONTEXT_MESSAGES default is 4
    assert "user: a" not in key
    assert "assistant: b" in key
    assert "user: e" in key


def test_openai_shape_build_response():
    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    resp = adapter.build_response("prompt", "answer")
    assert resp["id"] == "chatcmpl-cached"
    assert resp["choices"][0]["message"]["content"] == "answer"


def test_openai_shape_extract_response_text_and_tokens():
    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    raw = {"choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 42}}
    assert adapter.extract_response_text(raw) == "answer"
    assert adapter.extract_token_usage(raw) == 42


@patch("llm_cache_gateway.adapters.openai_shape.httpx.post")
def test_openai_shape_forward_request_sends_bearer_auth(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_post.return_value = mock_response

    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    result = adapter.forward_request({"model": "x"}, "key123")

    assert result == {"ok": True}
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer key123"


def test_openai_shape_stream_text_extraction():
    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    assert adapter.extract_stream_text('data: {"choices":[{"delta":{"content":"hi"}}]}') == "hi"
    assert adapter.extract_stream_text("data: [DONE]") is None


def test_openai_shape_format_stream_chunk_roundtrips():
    adapter = OpenAIShapeAdapter(base_url="https://example.com")
    line = adapter.format_stream_chunk("hello")
    assert adapter.extract_stream_text(line) == "hello"


def test_anthropic_extract_prompt_handles_block_content():
    adapter = AnthropicShapeAdapter()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    assert adapter.extract_prompt(body) == "hello"


def test_anthropic_extract_token_usage_sums_input_and_output():
    adapter = AnthropicShapeAdapter()
    raw = {"usage": {"input_tokens": 10, "output_tokens": 5}}
    assert adapter.extract_token_usage(raw) == 15


@patch("llm_cache_gateway.adapters.anthropic_shape.httpx.post")
def test_anthropic_forward_request_sends_x_api_key(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_post.return_value = mock_response

    adapter = AnthropicShapeAdapter()
    adapter.forward_request({"model": "x"}, "key123")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["x-api-key"] == "key123"
    assert "anthropic-version" in kwargs["headers"]


def test_gemini_extract_prompt_joins_multiple_text_parts():
    adapter = GeminiShapeAdapter()
    body = {"contents": [{"role": "user", "parts": [{"text": "hi "}, {"text": "there"}]}]}
    assert adapter.extract_prompt(body) == "hi there"


def test_gemini_forward_request_puts_model_in_url(monkeypatch=None):
    with patch("llm_cache_gateway.adapters.gemini_shape.httpx.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response

        adapter = GeminiShapeAdapter()
        adapter.forward_request({"model": "gemini-x", "contents": []}, "key123")

        args, kwargs = mock_post.call_args
        assert args[0].endswith("models/gemini-x:generateContent")
        assert kwargs["headers"]["x-goog-api-key"] == "key123"
        assert "model" not in kwargs["json"]


def test_cohere_extract_response_text():
    adapter = CohereShapeAdapter()
    raw = {"message": {"content": [{"text": "answer"}]}}
    assert adapter.extract_response_text(raw) == "answer"


@pytest.mark.parametrize("adapter", [AnthropicShapeAdapter(), GeminiShapeAdapter(), CohereShapeAdapter()])
def test_non_streaming_adapters_raise_not_implemented(adapter):
    assert adapter.supports_streaming is False
    with pytest.raises(NotImplementedError):
        adapter.forward_request_stream({}, "key")
    with pytest.raises(NotImplementedError):
        adapter.extract_stream_text("data: {}")
    with pytest.raises(NotImplementedError):
        adapter.format_stream_chunk("text")
