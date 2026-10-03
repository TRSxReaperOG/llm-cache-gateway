from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "llm_cache"
VECTOR_SIZE = 1024

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
