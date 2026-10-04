from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llm_cache_gateway import db
from llm_cache_gateway.main import ADAPTERS, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_collection():
    db.ensure_collection()
    db.clear_all()
    yield
    db.clear_all()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unknown_provider_returns_404():
    resp = client.post("/v1/proxy/nope/chat/completions", headers={"Authorization": "Bearer x"}, json={})
    assert resp.status_code == 404


def test_miss_then_hit_full_lifecycle():
    canned = {"choices": [{"message": {"content": "Paris"}}], "usage": {"total_tokens": 5}}
    body = {"model": "x", "messages": [{"role": "user", "content": "What's the capital of France?"}]}

    with patch.object(ADAPTERS["groq"], "forward_request", return_value=canned) as mock_forward:
        resp1 = client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body)
        assert resp1.status_code == 200
        assert resp1.json()["choices"][0]["message"]["content"] == "Paris"

        resp2 = client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body)
        assert resp2.status_code == 200
        assert resp2.json()["id"] == "chatcmpl-cached"
        assert resp2.json()["choices"][0]["message"]["content"] == "Paris"

        mock_forward.assert_called_once()  # second request was served from cache, not a real call


def test_admin_stats_and_entries_reflect_stored_data():
    canned = {"choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 3}}
    body = {"model": "x", "messages": [{"role": "user", "content": "a unique question"}]}

    with patch.object(ADAPTERS["groq"], "forward_request", return_value=canned):
        client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body)

    stats = client.get("/admin/stats").json()
    assert stats["total_entries"] == 1

    entries = client.get("/admin/entries").json()
    assert len(entries) == 1
    assert entries[0]["response"] == "answer"


def test_admin_clear_cache_empties_collection():
    canned = {"choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 3}}
    body = {"model": "x", "messages": [{"role": "user", "content": "another question"}]}

    with patch.object(ADAPTERS["groq"], "forward_request", return_value=canned):
        client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body)

    resp = client.delete("/admin/cache")
    assert resp.status_code == 200
    assert client.get("/admin/stats").json()["total_entries"] == 0


def test_admin_delete_single_entry():
    canned = {"choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 3}}
    body = {"model": "x", "messages": [{"role": "user", "content": "yet another question"}]}

    with patch.object(ADAPTERS["groq"], "forward_request", return_value=canned):
        client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body)

    entry_id = client.get("/admin/entries").json()[0]["id"]
    resp = client.delete(f"/admin/cache/{entry_id}")
    assert resp.status_code == 200
    assert client.get("/admin/stats").json()["total_entries"] == 0


def test_streaming_unsupported_provider_returns_501():
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/proxy/anthropic/chat/completions", headers={"Authorization": "Bearer x"}, json=body)
    assert resp.status_code == 501


def test_multiturn_context_prevents_false_cache_hit():
    canned_a = {"choices": [{"message": {"content": "Japan trains answer"}}], "usage": {"total_tokens": 5}}
    canned_b = {"choices": [{"message": {"content": "France trains answer"}}], "usage": {"total_tokens": 5}}

    body_a = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "Tell me about Japan."},
            {"role": "assistant", "content": "Japan is a country in East Asia."},
            {"role": "user", "content": "What about trains?"},
        ],
    }
    body_b = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "Tell me about France."},
            {"role": "assistant", "content": "France is a country in Europe."},
            {"role": "user", "content": "What about trains?"},
        ],
    }

    with patch.object(ADAPTERS["groq"], "forward_request", side_effect=[canned_a, canned_b]) as mock_forward:
        resp_a = client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body_a)
        resp_b = client.post("/v1/proxy/groq/chat/completions", headers={"Authorization": "Bearer x"}, json=body_b)

    assert resp_a.json()["choices"][0]["message"]["content"] == "Japan trains answer"
    assert resp_b.json()["choices"][0]["message"]["content"] == "France trains answer"
    assert mock_forward.call_count == 2  # both were real calls, no false cache collision
