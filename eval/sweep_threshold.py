import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(EVAL_DIR.parent / "src"))

from llm_cache_gateway.config import EMBEDDING_BACKEND  # noqa: E402
from llm_cache_gateway.embeddings import embed_texts  # noqa: E402

COLORS = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}
THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(30)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


dataset = json.loads((EVAL_DIR / "dataset.json").read_text())
unique_prompts = sorted({pair["prompt_a"] for pair in dataset} | {pair["prompt_b"] for pair in dataset})

print(f"Embedding {len(unique_prompts)} unique prompts via {EMBEDDING_BACKEND} backend...")
vectors = embed_texts(unique_prompts)
vector_by_prompt = dict(zip(unique_prompts, vectors))

pairs = [
    (pair["should_match"], cosine_similarity(vector_by_prompt[pair["prompt_a"]], vector_by_prompt[pair["prompt_b"]]))
    for pair in dataset
]

rows = []
for threshold in THRESHOLDS:
    tp = sum(1 for should_match, sim in pairs if should_match and sim >= threshold)
    fp = sum(1 for should_match, sim in pairs if not should_match and sim >= threshold)
    fn = sum(1 for should_match, sim in pairs if should_match and sim < threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})

csv_path = RESULTS_DIR / f"{EMBEDDING_BACKEND}_sweep.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {csv_path}")

best_f1_row = max(rows, key=lambda r: r["f1"])
zero_fp_rows = [r for r in rows if r["fp"] == 0]
best_zero_fp_row = min(zero_fp_rows, key=lambda r: r["threshold"]) if zero_fp_rows else None

print(f"Best F1: {best_f1_row['f1']:.3f} at threshold {best_f1_row['threshold']}")
if best_zero_fp_row:
    print(f"Lowest zero-false-positive threshold: {best_zero_fp_row['threshold']} (recall {best_zero_fp_row['recall']:.3f})")
else:
    print("No threshold in the sweep range achieves zero false positives.")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
for metric in ("precision", "recall", "f1"):
    ax.plot(
        [r["threshold"] for r in rows],
        [r[metric] for r in rows],
        label=metric.capitalize(),
        color=COLORS[metric],
        linewidth=2,
    )
ax.set_xlabel("Similarity threshold", color="#0b0b0b")
ax.set_ylabel("Score", color="#0b0b0b")
ax.set_title(f"Precision / Recall / F1 vs. threshold ({EMBEDDING_BACKEND} embeddings)", color="#0b0b0b")
ax.set_ylim(-0.02, 1.02)
ax.grid(True, color="#e1e0d9", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#c3c2b7")
ax.tick_params(colors="#898781")
ax.legend(frameon=False)
fig.tight_layout()

png_path = RESULTS_DIR / f"{EMBEDDING_BACKEND}_threshold_curve.png"
fig.savefig(png_path, dpi=150)
print(f"Wrote {png_path}")
