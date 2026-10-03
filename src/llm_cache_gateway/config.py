import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "voyage")
