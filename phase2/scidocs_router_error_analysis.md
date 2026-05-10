# SCIDOCS Router Error Analysis

## Scope Note

This analysis refers to the archived default SCIDOCS router snapshot:

- weak-label rule: `MRR@10`, tie -> `hybrid`
- retrieval-confidence features: off

That historical snapshot is useful because it explains why the original SCIDOCS router underperformed fixed `dense`.

It is not the current best SCIDOCS router configuration anymore.

The later margin-aware plus retrieval-confidence variant is documented separately in [phase2/scidocs_margin_retrieval_comparison.md](scidocs_margin_retrieval_comparison.md) and now slightly beats fixed `dense` on the SCIDOCS self-split setting.

## Question

Why did the archived default router lose to the fixed `dense` baseline on SCIDOCS?

Current Phase 1 snapshot on the SCIDOCS self-split test set:

- `dense` MRR@10: `0.3723`
- `router` MRR@10: `0.3620`
- gap: `-0.0102`

So the router is close, but still behind `dense`.

## Main Finding

The biggest problem is not just model capacity. The larger issue is that the SCIDOCS supervision signal is biased toward `hybrid`, especially because of the current tie-breaking rule:

- labels are assigned with `MRR@10`
- when two modes tie, `best_mode_for_query(...)` prefers `hybrid`

That matters a lot on SCIDOCS because many queries have the same `MRR@10` under `dense` and `hybrid`.

## Key Numbers

From the current SCIDOCS router test set:

- total test queries: `200`
- router average MRR@10: `0.3620`
- dense average MRR@10: `0.3723`
- average gap: `-0.0102`

Label and prediction distribution:

- gold labels: `hybrid=131`, `dense=52`, `sparse=17`
- router predictions: `hybrid=119`, `dense=73`, `sparse=8`

The gold labels are already strongly skewed toward `hybrid`.

## Most Important Structural Issue

Dense-hybrid tie frequency is extremely high:

- queries where `dense` and `hybrid` tie at the top on MRR@10: `100 / 200`
- all-zero ties among all three modes: `55 / 200`

This means half of the SCIDOCS test queries do not actually show a meaningful `dense` vs `hybrid` quality difference under the current label metric.

Because the code currently breaks ties in favor of `hybrid`, many ambiguous queries become `hybrid` labels even when `dense` is equally good.

This creates two downstream problems:

1. the router is trained to predict `hybrid` more often than true retrieval quality justifies
2. when the router predicts `dense` on a tied query, it can look “wrong” in classification terms even though it is not worse in ranking quality

## Where The Router Really Loses To Dense

The main loss does not come from all misclassifications equally. It comes from one specific error pattern:

- `gold=dense`, `pred=hybrid`: `30` queries
- `gold=dense`, `pred=sparse`: `3` queries
- total dense-gold misrouted queries: `33`
- average MRR delta versus dense on those `33` queries: `-0.3122`

This is the dominant failure mode.

By contrast:

- `gold=hybrid`, `pred=dense`: `49` queries
- average MRR delta versus dense on those queries: `0.0`

That second number is extremely important. It means many queries labeled as `hybrid` do not actually hurt ranking quality when the router sends them to `dense`.

So the router’s biggest real mistake is:

- missing true dense-friendly queries and sending them to `hybrid`

Its apparent mistake:

- sending many labeled-`hybrid` queries to `dense`

is often not a retrieval-quality mistake at all.

## Worst Loss Examples

Some of the largest router losses against dense come from queries like:

- `Link Prediction using Supervised Learning *`
- `Generate to Adapt: Aligning Domains Using Generative Adversarial Networks`
- `Automatic ranking of swear words using word embeddings and pseudo-relevance feedback`
- `nTorrent: Peer-to-Peer File Sharing in Named Data Networking`
- `Distributed Learning over Unreliable Networks`

These are cases where the router predicted `hybrid`, but `dense` was clearly better.

## Query Pattern Analysis

### Dense-Gold Queries Misrouted Away From Dense

For the `33` dense-gold mistakes:

- average token length: `8.64`
- average character length: `65.52`
- average acronym ratio: `0.0692`
- queries containing hyphens: `39.4%`
- average uppercase ratio: `0.1354`

Compared with dense-gold queries correctly sent to dense:

- average token length: `9.74`
- average character length: `72.11`
- average acronym ratio: `0.0066`
- queries containing hyphens: `42.1%`
- average uppercase ratio: `0.0744`

Interpretation:

- the router is especially shaky on acronym-heavy and uppercase-heavy dense-friendly queries
- those queries often look “technical” or “keyword-like”
- the current query-only features seem to over-associate those patterns with `hybrid`

Common terms among dense-gold mistakes include:

- `using`
- `networks`
- `learning`
- `domain`
- `face`
- `model`
- `user`
- `social`

These are broad technical terms that may look ambiguous from query text alone, but dense retrieval often handles them well.

### Hybrid-Gold Queries Sent To Dense

For the `49` hybrid-gold queries that the router sent to dense:

- average MRR delta versus dense: `0.0`

So this category mostly does not explain the ranking gap. In other words:

- these are mostly classification errors
- not actual retrieval-quality errors

This strongly suggests the current SCIDOCS label space is noisier than the ranking gap alone implies.

## Root Causes

The SCIDOCS router loses to dense mainly for three reasons.

### 1. Tie-biased supervision

The training labels over-favor `hybrid` because ties are broken toward `hybrid`.

Since `dense` and `hybrid` tie very often on SCIDOCS, this makes the label distribution more hybrid-heavy than the real quality difference supports.

### 2. Query-only features are not enough for SCIDOCS

The current router mostly sees text shape and lexical patterns. That is enough to help on SciFact, but on SCIDOCS many dense-vs-hybrid decisions are too subtle to recover from query text alone.

The router needs retrieval-side evidence such as:

- dense top-score confidence
- dense-vs-hybrid overlap
- score margin between rank 1 and rank 2
- agreement between sparse and dense top documents

### 3. The wrong mistakes are expensive

The router can afford some `hybrid -> dense` classification disagreement when the ranking outcome is tied.

It cannot afford many `dense -> hybrid` mistakes on queries where dense is strictly better.

That is exactly where most of the real loss is concentrated.

## Practical Conclusion

The current SCIDOCS router is not fundamentally failing because it never learns anything useful. It is failing because:

- the supervision is noisy and hybrid-biased
- dense and hybrid tie too often under the current metric
- the router still lacks strong signals to separate truly dense-favored queries from merely ambiguous ones

So the most important next improvements are:

1. change tie handling during weak labeling
2. add retrieval-confidence features
3. use cost-aware or quality-margin-aware labeling rather than hard `MRR@10` winner-take-all labels

## Recommended Next Fixes

### Option 1: Fix the label rule

Instead of always preferring `hybrid` on ties:

- prefer `dense` on ties when quality is equal and latency is lower
- or label ties as a separate ambiguous class during analysis
- or skip low-information tied examples during training

This is likely the single highest-value fix for SCIDOCS.

### Option 2: Add margin-aware labeling

Only assign `hybrid` when it beats `dense` by a real margin.

For example:

- if `hybrid_mrr - dense_mrr < epsilon`, label the query as `dense`

That would reduce supervision noise substantially.

### Option 3: Add retrieval-side router features

The next router version should include:

- dense top-1 score
- dense top-1 vs top-2 gap
- sparse/dense top-k overlap
- whether dense and hybrid return the same top document

These should help distinguish real dense wins from noisy hybrid labels.

## Summary

SCIDOCS does not mainly fail because the router is too simple. It mainly fails because the current weak-label construction pushes too many ambiguous queries into the `hybrid` class, while the real ranking loss comes from a smaller set of truly dense-favored queries that the router misses.

So if we want the router to beat dense on SCIDOCS, the highest-priority fix is:

- clean up the weak supervision rule before making the classifier much more complex
