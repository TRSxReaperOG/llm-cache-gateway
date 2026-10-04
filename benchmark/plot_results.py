import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
sys.path.insert(0, str(BENCHMARK_DIR.parent / "src"))

from llm_cache_gateway.pricing import estimate_cost  # noqa: E402

COLORS = {"line": "#2a78d6", "spent": "#eb6834", "no_cache": "#1baf7a", "hit": "#2a78d6", "miss": "#eb6834"}
PROVIDER = "groq"

with (RESULTS_DIR / "load_test_log.csv").open() as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["hit"] = r["hit"] == "True"
    r["latency_ms"] = float(r["latency_ms"])
    r["tokens"] = int(r["tokens"])
    r["request_index"] = int(r["request_index"])


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")


miss_tokens = [r["tokens"] for r in rows if not r["hit"] and r["tokens"] > 0]
avg_miss_tokens = sum(miss_tokens) / len(miss_tokens) if miss_tokens else 0
overall_hit_rate = sum(1 for r in rows if r["hit"]) / len(rows)

# --- Chart 1: rolling hit rate over time ---
window = 15
hit_rates = []
for i in range(len(rows)):
    chunk = rows[max(0, i - window + 1) : i + 1]
    hit_rates.append(sum(1 for r in chunk if r["hit"]) / len(chunk))

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
style(ax)
ax.plot([r["request_index"] for r in rows], hit_rates, color=COLORS["line"], linewidth=2)
ax.set_xlabel("Request #", color="#0b0b0b")
ax.set_ylabel(f"Rolling hit rate (window={window})", color="#0b0b0b")
ax.set_title(f"Cache hit rate over the load test (overall: {overall_hit_rate:.0%})", color="#0b0b0b")
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "hit_rate_over_time.png", dpi=150)
print("Wrote hit_rate_over_time.png")

# --- Chart 2: cumulative cost spent vs. estimated no-cache cost ---
spent = 0.0
no_cache = 0.0
cumulative_spent = []
cumulative_no_cache = []
for r in rows:
    if r["hit"]:
        cumulative_this_spend = 0.0
        avoided_cost = estimate_cost(PROVIDER, avg_miss_tokens)
    else:
        cumulative_this_spend = estimate_cost(PROVIDER, r["tokens"])
        avoided_cost = cumulative_this_spend
    spent += cumulative_this_spend
    no_cache += avoided_cost
    cumulative_spent.append(spent)
    cumulative_no_cache.append(no_cache)

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
style(ax)
ax.plot(
    [r["request_index"] for r in rows],
    cumulative_no_cache,
    label="Without cache (estimated)",
    color=COLORS["no_cache"],
    linewidth=2,
)
ax.plot(
    [r["request_index"] for r in rows], cumulative_spent, label="With cache (actual)", color=COLORS["spent"], linewidth=2
)
ax.set_xlabel("Request #", color="#0b0b0b")
ax.set_ylabel("Cumulative cost (USD, estimated)", color="#0b0b0b")
total_saved = cumulative_no_cache[-1] - cumulative_spent[-1]
pct_saved = 100 * total_saved / cumulative_no_cache[-1] if cumulative_no_cache[-1] else 0
ax.set_title(f"Cost saved by caching (~${total_saved:.4f}, {pct_saved:.0f}% of no-cache cost)", color="#0b0b0b")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cost_saved.png", dpi=150)
print("Wrote cost_saved.png")
print(f"Total estimated savings: ${total_saved:.4f} ({pct_saved:.1f}% of no-cache cost)")

# --- Chart 3: latency, hit vs. miss ---
hit_latencies = [r["latency_ms"] for r in rows if r["hit"]]
miss_latencies = [r["latency_ms"] for r in rows if not r["hit"]]

fig, ax = plt.subplots(figsize=(6, 5), facecolor="#fcfcfb")
style(ax)
bp = ax.boxplot(
    [miss_latencies, hit_latencies], tick_labels=["Miss (real call)", "Hit (cached)"], patch_artist=True, widths=0.5
)
for patch, color in zip(bp["boxes"], [COLORS["miss"], COLORS["hit"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for element in ("whiskers", "caps", "medians"):
    for item in bp[element]:
        item.set_color("#52514e")
ax.set_ylabel("Latency (ms)", color="#0b0b0b")
ax.set_title("Latency: cache hit vs. real call", color="#0b0b0b")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "latency_hit_vs_miss.png", dpi=150)
print("Wrote latency_hit_vs_miss.png")

hit_median = sorted(hit_latencies)[len(hit_latencies) // 2] if hit_latencies else None
miss_median = sorted(miss_latencies)[len(miss_latencies) // 2] if miss_latencies else None
print(f"Hit median latency: {hit_median:.1f}ms | Miss median latency: {miss_median:.1f}ms")
if hit_median:
    print(f"Hits were ~{miss_median / hit_median:.1f}x faster than misses")
