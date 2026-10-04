import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "voyage")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
TTL_SECONDS = int(os.environ.get("TTL_SECONDS", 86400))
MAX_CACHE_SIZE = int(os.environ.get("MAX_CACHE_SIZE", 10000))
CONTEXT_MESSAGES = int(os.environ.get("CONTEXT_MESSAGES", 4))
