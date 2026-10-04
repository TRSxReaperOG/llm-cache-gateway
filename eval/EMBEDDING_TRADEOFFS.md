# Embedding tradeoffs: Voyage vs. local (all-MiniLM-L6-v2)

Measured on the same 30-pair eval set (`dataset.json`), same methodology as
`THRESHOLD_DECISION.md`. Full curves in `results/voyage_threshold_curve.png`,
`results/local_threshold_curve.png`, and `results/comparison_f1.png`.

## Quality

| | Best F1 | Threshold | Zero-FP threshold | Zero-FP recall |
|---|---|---|---|---|
| Voyage (voyage-3.5-lite) | 0.867 | 0.88 | 0.93 | 0.267 |
| Local (all-MiniLM-L6-v2) | 0.897 | 0.84 | 0.93 | 0.133 |

Local embeddings scored a slightly *higher* peak F1 than Voyage on this set,
which runs against the intuition that a paid, larger embedding API should
outperform a small local model. Plausible reason: all-MiniLM-L6-v2 was trained
specifically on sentence-similarity (STS) objectives, which is exactly the task
this eval measures, while Voyage's model is tuned more broadly for retrieval
across domains. That doesn't make local "better" in general — it means the two
models calibrate cosine similarity differently, and a threshold tuned for one
doesn't necessarily transfer to the other (though in this case, both backends'
zero-false-positive points happened to land at the same threshold, 0.93 —
convenient, but not something to assume holds in general; re-check if either
model is swapped).

At the zero-false-positive point specifically, Voyage retains roughly 2x the
recall of local (0.267 vs. 0.133) — Voyage separates true paraphrases from
confusable pairs more cleanly at the conservative end of the curve, even though
its peak F1 is lower.

## Cost & latency (not measured here — qualitative)

- **Voyage**: network round-trip per request (or per batch), usage-metered API
  cost, external dependency (down → cache degrades, which is exactly why the
  local fallback exists).
- **Local (all-MiniLM-L6-v2)**: no network call, no per-token cost, runs on-box;
  tradeoff is it competes with the rest of the process for CPU, and the model
  has to be loaded into memory at startup (small — 384-dim, ~80MB model).

## Takeaway

Given comparable measured quality and the local model's zero marginal cost and
no external dependency, the local fallback is a credible primary candidate too,
not just an emergency fallback — worth a deliberate cost/latency benchmark in a
later phase before deciding whether Voyage is actually pulling its weight as
the default.
