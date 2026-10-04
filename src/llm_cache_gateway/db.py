import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Direction,
    Distance,
    FieldCondition,
    Filter,
    OrderBy,
    PayloadSchemaType,
    PointStruct,
    Range,
    ScoredPoint,
    VectorParams,
)

from llm_cache_gateway.config import EMBEDDING_BACKEND, MAX_CACHE_SIZE, QDRANT_URL, TTL_SECONDS
from llm_cache_gateway.embeddings import EMBED_DIMENSION

COLLECTION_NAME = f"llm_cache_{EMBEDDING_BACKEND}"
VECTOR_SIZE = EMBED_DIMENSION

client = QdrantClient(location=QDRANT_URL)


REQUIRED_INDEXES = {
    "provider": PayloadSchemaType.KEYWORD,
    "timestamp": PayloadSchemaType.FLOAT,
    "last_accessed": PayloadSchemaType.FLOAT,
}


def ensure_collection() -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    existing_indexes = client.get_collection(COLLECTION_NAME).payload_schema
    for field_name, schema in REQUIRED_INDEXES.items():
        if field_name not in existing_indexes:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )


def _not_expired_filter() -> Filter:
    cutoff = time.time() - TTL_SECONDS
    return Filter(must=[FieldCondition(key="timestamp", range=Range(gte=cutoff))])


def search(vector: list[float]) -> list[ScoredPoint]:
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=_not_expired_filter(),
        limit=1,
        with_payload=True,
    )
    return result.points


def record_hit(point_id: str) -> None:
    points = client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id], with_payload=["hit_count"])
    hit_count = (points[0].payload or {}).get("hit_count", 0) if points else 0

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"hit_count": hit_count + 1, "last_accessed": time.time()},
        points=[point_id],
    )


def store(vector: list[float], prompt: str, response: str, provider: str, tokens: int) -> None:
    now = time.time()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "prompt": prompt,
            "response": response,
            "timestamp": now,
            "last_accessed": now,
            "hit_count": 0,
            "provider": provider,
            "tokens": tokens,
        },
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    purge_expired()
    _evict_if_over_limit()


def purge_expired() -> int:
    cutoff = time.time() - TTL_SECONDS
    expired_filter = Filter(must=[FieldCondition(key="timestamp", range=Range(lt=cutoff))])
    before = client.count(COLLECTION_NAME, exact=True).count
    client.delete(collection_name=COLLECTION_NAME, points_selector=expired_filter)
    after = client.count(COLLECTION_NAME, exact=True).count
    return before - after


def _evict_if_over_limit() -> None:
    total = client.count(COLLECTION_NAME, exact=True).count
    excess = total - MAX_CACHE_SIZE
    if excess <= 0:
        return

    oldest, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=excess,
        order_by=OrderBy(key="last_accessed", direction=Direction.ASC),
        with_payload=False,
        with_vectors=False,
    )
    ids = [point.id for point in oldest]
    if ids:
        client.delete(collection_name=COLLECTION_NAME, points_selector=ids)


def get_stats() -> dict:
    total = client.count(COLLECTION_NAME, exact=True).count

    total_hits = 0
    per_provider: dict[str, int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["provider", "hit_count"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            total_hits += payload.get("hit_count", 0)
            provider = payload.get("provider", "unknown")
            per_provider[provider] = per_provider.get(provider, 0) + 1
        if offset is None:
            break

    return {"total_entries": total, "total_hits": total_hits, "entries_per_provider": per_provider}


def list_recent(limit: int = 20) -> list[dict]:
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        order_by=OrderBy(key="timestamp", direction=Direction.DESC),
        with_payload=True,
        with_vectors=False,
    )
    return [{"id": point.id, **(point.payload or {})} for point in points]


def delete_entry(point_id: str) -> None:
    client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])


def clear_all() -> None:
    client.delete_collection(COLLECTION_NAME)
    ensure_collection()
