import os

from llm_cache_gateway.config import EMBEDDING_BACKEND

VOYAGE_MODEL = "voyage-3.5-lite"
VOYAGE_DIMENSION = 1024

LOCAL_MODEL = "all-MiniLM-L6-v2"
LOCAL_DIMENSION = 384

if EMBEDDING_BACKEND == "voyage":
    import voyageai

    EMBED_DIMENSION = VOYAGE_DIMENSION
    _voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

elif EMBEDDING_BACKEND == "local":
    from sentence_transformers import SentenceTransformer

    EMBED_DIMENSION = LOCAL_DIMENSION
    _local_model = SentenceTransformer(LOCAL_MODEL)

else:
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {EMBEDDING_BACKEND!r}")


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_BACKEND == "voyage":
        result = _voyage_client.embed(
            texts,
            model=VOYAGE_MODEL,
            output_dimension=VOYAGE_DIMENSION,
        )
        return result.embeddings

    return _local_model.encode(texts).tolist()
