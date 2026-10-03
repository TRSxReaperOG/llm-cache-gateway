import os

import voyageai
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "voyage-3.5-lite"
EMBED_DIMENSION = 1024

_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])


def embed_text(text: str) -> list[float]:
    result = _client.embed(
        [text],
        model=EMBED_MODEL,
        output_dimension=EMBED_DIMENSION,
    )
    return result.embeddings[0]
