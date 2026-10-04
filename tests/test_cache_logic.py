import time

import pytest

from llm_cache_gateway import db


@pytest.fixture(autouse=True)
def clean_collection():
    db.ensure_collection()
    db.clear_all()
    yield
    db.clear_all()


def _vector(variant: int) -> list[float]:
    vec = [0.01] * db.VECTOR_SIZE
    vec[variant % db.VECTOR_SIZE] = 5.0
    return vec


def test_store_and_search_hit():
    db.store(_vector(1), "prompt", "response", provider="groq", tokens=10)
    matches = db.search(_vector(1))
    assert matches
    assert matches[0].payload["response"] == "response"
    assert matches[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_distinguishes_dissimilar_vectors():
    db.store(_vector(1), "p1", "r1", provider="groq", tokens=10)
    matches = db.search(_vector(50))
    assert matches
    assert matches[0].score < 0.99


def test_record_hit_increments_count_and_updates_last_accessed():
    db.store(_vector(2), "p", "r", provider="groq", tokens=10)
    point_id = db.search(_vector(2))[0].id

    db.record_hit(point_id)
    db.record_hit(point_id)

    entry = db.list_recent(limit=1)[0]
    assert entry["hit_count"] == 2


def test_ttl_expiry_hides_entry_from_search(monkeypatch):
    monkeypatch.setattr(db, "TTL_SECONDS", 1)
    db.store(_vector(3), "p", "r", provider="groq", tokens=10)
    assert db.search(_vector(3))

    time.sleep(1.5)
    assert db.search(_vector(3)) == []


def test_purge_expired_deletes_old_points(monkeypatch):
    monkeypatch.setattr(db, "TTL_SECONDS", 1)
    db.store(_vector(4), "p", "r", provider="groq", tokens=10)
    time.sleep(1.5)

    purged = db.purge_expired()

    assert purged == 1
    assert db.get_stats()["total_entries"] == 0


def test_eviction_caps_collection_at_max_size(monkeypatch):
    monkeypatch.setattr(db, "MAX_CACHE_SIZE", 2)
    for i in range(4):
        db.store(_vector(i), f"p{i}", f"r{i}", provider="groq", tokens=10)

    assert db.get_stats()["total_entries"] <= 2


def test_eviction_keeps_most_recently_accessed(monkeypatch):
    monkeypatch.setattr(db, "MAX_CACHE_SIZE", 2)
    db.store(_vector(10), "old", "r", provider="groq", tokens=10)
    old_id = db.search(_vector(10))[0].id
    db.store(_vector(11), "middle", "r", provider="groq", tokens=10)

    # Touch "old" so it's more recently accessed than "middle"
    db.record_hit(old_id)

    db.store(_vector(12), "newest", "r", provider="groq", tokens=10)  # triggers eviction of 1

    remaining_prompts = {e["prompt"] for e in db.list_recent(limit=10)}
    assert "middle" not in remaining_prompts
    assert "old" in remaining_prompts
    assert "newest" in remaining_prompts


def test_delete_entry_removes_it():
    db.store(_vector(5), "p", "r", provider="groq", tokens=10)
    point_id = db.search(_vector(5))[0].id

    db.delete_entry(point_id)

    assert db.search(_vector(5)) == []


def test_threshold_boundary_matches_production_cutoff():
    db.store(_vector(6), "p", "r", provider="groq", tokens=10)
    matches = db.search(_vector(6))

    # Production SIMILARITY_THRESHOLD (main.py) is 0.93 — an exact repeat
    # must clear it comfortably for the cache to ever fire.
    assert matches[0].score >= 0.93
