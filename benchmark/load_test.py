import csv
import os
import random
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

BENCHMARK_DIR = Path(__file__).parent
load_dotenv(BENCHMARK_DIR.parent / ".env")
sys.path.insert(0, str(BENCHMARK_DIR))

from traffic_prompts import PROMPT_POOL  # noqa: E402

GATEWAY_URL = "http://localhost:8000/v1/proxy/groq/chat/completions"
MODEL = "openai/gpt-oss-20b"
NUM_REQUESTS = 150
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)  # reproducible traffic pattern across runs

api_key = os.environ["GROQ_API_KEY"]


def generate_traffic(num_requests: int) -> list[tuple[str, str]]:
    used_base_indices: list[int] = []
    traffic = []
    for _ in range(num_requests):
        roll = random.random()
        if used_base_indices and roll < 0.4:
            idx = random.choice(used_base_indices)
            traffic.append((PROMPT_POOL[idx]["base"], "repeat"))
        elif used_base_indices and roll < 0.7:
            idx = random.choice(used_base_indices)
            traffic.append((random.choice(PROMPT_POOL[idx]["paraphrases"]), "paraphrase"))
        else:
            remaining = [j for j in range(len(PROMPT_POOL)) if j not in used_base_indices]
            idx = random.choice(remaining) if remaining else random.choice(range(len(PROMPT_POOL)))
            if remaining:
                used_base_indices.append(idx)
            traffic.append((PROMPT_POOL[idx]["base"], "novel"))
    return traffic


def run() -> None:
    traffic = generate_traffic(NUM_REQUESTS)
    rows = []

    for i, (text, kind) in enumerate(traffic, 1):
        body = {"model": MODEL, "messages": [{"role": "user", "content": text}]}
        start = time.monotonic()
        while True:
            resp = httpx.post(GATEWAY_URL, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0)
            # The gateway doesn't propagate upstream 429s cleanly yet (known gap,
            # surfaces as a 500) — retry with a fixed backoff rather than reading
            # a retry-after header that won't be there.
            if resp.status_code >= 500:
                print("  gateway error (likely upstream rate limit), waiting 10s")
                time.sleep(10)
                start = time.monotonic()  # don't count the wait as request latency
                continue
            break
        latency_ms = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        is_hit = data.get("id") == "chatcmpl-cached"
        tokens = data.get("usage", {}).get("total_tokens", 0)

        rows.append(
            {
                "request_index": i,
                "prompt_type": kind,
                "hit": is_hit,
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens,
            }
        )
        print(f"[{i}/{NUM_REQUESTS}] {kind:10s} {'HIT ' if is_hit else 'MISS'} {latency_ms:7.1f}ms  {text[:50]}")

    csv_path = RESULTS_DIR / "load_test_log.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    run()
