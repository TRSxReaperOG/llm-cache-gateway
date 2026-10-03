import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, PointStruct, ScoredPoint, VectorParams

from llm_cache_gateway.config import EMBEDDING_BACKEND
from llm_cache_gateway.embeddings import EMBED_DIMENSION

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = f"llm_cache_{EMBEDDING_BACKEND}"
VECTOR_SIZE = EMBED_DIMENSION

client = QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="provider",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def search(vector: list[float]) -> list[ScoredPoint]:
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1,
        with_payload=True,
    )
    return result.points


def store(vector: list[float], prompt: str, response: str, provider: str, tokens: int) -> None:
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "provider": provider,
            "tokens": tokens,
        },
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
