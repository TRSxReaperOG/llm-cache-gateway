import os

# Must be set before any llm_cache_gateway module is imported anywhere in the
# test session: config.py and embeddings.py read these at import time.
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("TTL_SECONDS", "86400")
os.environ.setdefault("MAX_CACHE_SIZE", "10000")
