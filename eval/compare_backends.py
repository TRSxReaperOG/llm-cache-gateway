import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
COLORS = {"voyage": "#2a78d6", "local": "#eb6834"}


def load_rows(backend: str) -> list[dict]:
    with (RESULTS_DIR / f"{backend}_sweep.csv").open() as f:
        return [
            {k: (float(v) if k != "threshold" else float(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")

for backend in ("voyage", "local"):
    rows = load_rows(backend)
    ax.plot(
        [r["threshold"] for r in rows],
        [r["f1"] for r in rows],
        label=f"{backend.capitalize()} F1",
        color=COLORS[backend],
        linewidth=2,
    )
    best = max(rows, key=lambda r: r["f1"])
    print(f"{backend}: best F1 {best['f1']:.3f} at threshold {best['threshold']}")

ax.set_xlabel("Similarity threshold", color="#0b0b0b")
ax.set_ylabel("F1", color="#0b0b0b")
ax.set_title("Voyage vs. local embeddings: F1 vs. threshold", color="#0b0b0b")
ax.set_ylim(-0.02, 1.02)
ax.grid(True, color="#e1e0d9", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#c3c2b7")
ax.tick_params(colors="#898781")
ax.legend(frameon=False)
fig.tight_layout()

png_path = RESULTS_DIR / "comparison_f1.png"
fig.savefig(png_path, dpi=150)
print(f"Wrote {png_path}")
