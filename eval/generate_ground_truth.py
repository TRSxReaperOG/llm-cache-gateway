import json
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent
load_dotenv(EVAL_DIR.parent / ".env")
sys.path.insert(0, str(EVAL_DIR.parent / "src"))

import os  # noqa: E402

from llm_cache_gateway.adapters.openai_shape import OpenAIShapeAdapter  # noqa: E402

GROQ_MODEL = "openai/gpt-oss-20b"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"

adapter = OpenAIShapeAdapter(base_url="https://api.groq.com/openai/v1")
api_key = os.environ["GROQ_API_KEY"]

dataset = json.loads((EVAL_DIR / "dataset.json").read_text())
prompts = sorted({pair["prompt_a"] for pair in dataset} | {pair["prompt_b"] for pair in dataset})

ground_truth = json.loads(GROUND_TRUTH_PATH.read_text()) if GROUND_TRUTH_PATH.exists() else {}
remaining = [p for p in prompts if p not in ground_truth]

for i, prompt in enumerate(remaining, 1):
    print(f"[{i}/{len(remaining)}] {prompt}")
    body = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]}
    while True:
        try:
            response = adapter.forward_request(body, api_key)
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429:
                raise
            wait = float(e.response.headers.get("retry-after", 2))
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    ground_truth[prompt] = adapter.extract_response_text(response)
    GROUND_TRUTH_PATH.write_text(json.dumps(ground_truth, indent=2))

print(f"Wrote {len(ground_truth)} ground-truth answers to eval/ground_truth.json")
