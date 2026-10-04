# Similarity threshold decision

## Method

30 labeled prompt pairs (`dataset.json`): 15 true paraphrases (should cache-hit)
and 15 confusable-intent pairs that share a surface template but have different
correct answers (should NOT cache-hit — e.g. "capital of Portugal" vs. "capital
of Spain"). For each backend, embedded all 49 unique prompts, computed cosine
similarity per pair, swept thresholds 0.70→0.99 in steps of 0.01, and computed
precision/recall/F1 against the labels at each threshold (`sweep_threshold.py`,
results in `results/*_sweep.csv`, curves in `results/*_threshold_curve.png`).

## What the previous threshold (0.95) actually did

Never measured against data — picked without an eval set. Against the real
Voyage embeddings, 0.95 yields **precision 1.0, recall 0.2**: only 3 of 15 true
paraphrases would have been served from cache; the other 12 would all needlessly
re-hit the LLM. It was safe (no wrong answers) but far more conservative than
necessary.

## Selection rule

A false positive here means serving a wrong cached answer as if it were correct
(e.g. answering "capital of Portugal" with Spain's capital) — worse than a false
negative, which just costs one extra real LLM call. So the rule is: take the
threshold that maximizes F1, but only if it has zero false positives in this
eval set; otherwise fall back to the lowest threshold that does.

- Max F1 on Voyage is 0.867 at threshold **0.88** — but it has 2 false positives
  (precision 0.867), so it's rejected by the rule above.
- The lowest Voyage threshold with zero false positives is **0.93** (precision
  1.0, recall 0.267, catching 4/15 true paraphrases).

## Decision: SIMILARITY_THRESHOLD = 0.93

Applied in `main.py`. This is lower than the old 0.95 (so it catches one more
true paraphrase, 4/15 vs. 3/15, in this eval set) while still holding zero false
positives against all 15 confusable pairs tested.

Recall at this threshold is still low in absolute terms (0.267) — this eval set
deliberately uses creative/loose paraphrasing (different word choice, different
sentence structure) to stress-test the boundary. Real traffic duplicates (typos,
near-identical repeats, copy-pasted re-asks) sit much closer to cosine similarity
1.0 than these synthetic paraphrases do, so real-world recall should run higher
than this number suggests. This eval set measures the *safety boundary*, not
expected production hit rate — a follow-up with real traffic logs would be
needed to measure actual hit rate.

## Caveat

n=30 pairs is small; a single pair crossing a threshold shifts precision/recall
by 1/15 (~6.7 points). Treat these as directional, not statistically precise —
re-run with a larger eval set if the threshold ever needs re-justifying for a
higher-stakes deployment.
