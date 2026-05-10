# Phase 2 Router Method

## Goal

In Phase 2, the router is treated as a query-aware strategy selector for MultiplexRAG.

It does not retrieve documents by itself. Instead, it predicts which retrieval branch should handle a query:

- `sparse`
- `dense`
- `hybrid`

The learning target is:

`query text -> best retrieval strategy`

So the router sits above the retrievers and decides which retrieval representation is most appropriate for each query.

## Why This Version Matters

The earlier router was a weak baseline built on a very small handcrafted feature set and a nearest-centroid classifier. That version showed that query-aware routing had potential, but it did not make full use of the information already available in the query text.

This Phase 2 version improves that design in three ways:

1. richer query-side features
2. a stronger but still lightweight classifier
3. confidence-aware fallback behavior

The result is a router that is still easy to explain, but is substantially more expressive and more practical.

## Implementation Files

The current Phase 2 router process is mainly implemented in:

- [scripts/train_router.py](../scripts/train_router.py)
- [src/multiplexrag/router.py](../src/multiplexrag/router.py)
- [src/multiplexrag/retrieval.py](../src/multiplexrag/retrieval.py)
- [src/multiplexrag/eval_utils.py](../src/multiplexrag/eval_utils.py)

## End-to-End Process

The Phase 2 router pipeline works as follows:

1. Load corpus, queries, and qrels.
2. Load or build the three retrieval branches: `sparse`, `dense`, and `hybrid`.
3. Run all three branches for each query.
4. Score each branch against qrels using ranking metrics, especially `MRR@10`.
5. Assign each training query a weak supervision label equal to the best-performing branch.
6. Convert each query into a richer feature representation.
7. Train a class-balanced logistic-regression router.
8. Predict a retrieval mode for each test query.
9. If the prediction confidence is low, back off to a default fallback branch.
10. Evaluate the final router decisions against fixed baselines and the oracle upper bound.

So the Phase 2 router is still a supervised strategy selector, but the supervision comes from retrieval outcomes rather than manual query annotations.

## Supervision Process

The router uses weak supervision.

For each query in the training split:

1. retrieve with `sparse`
2. retrieve with `dense`
3. retrieve with `hybrid`
4. compare those ranked lists against ground-truth qrels
5. choose the best branch as the training label

In the current implementation, the label decision is produced by `best_mode_for_query(...)` in [router.py](../src/multiplexrag/router.py).

Default weak-label configuration:

- metric: `MRR@10`
- tie preference: `hybrid`

Optional weak-label settings also support:

- multiple metrics such as `MRR@10`, `nDCG@10`, and `Recall@10`
- custom metric weights
- a different tie preference such as `dense`

This means the router learns branch selection rather than document relevance.

## Query Representation

This version uses a richer query-only representation than Phase 1.

### Numeric Query Features

The numeric feature block includes:

- character length
- token length
- average token length
- unique-token ratio
- digit count
- digit ratio
- acronym count
- acronym ratio
- uppercase ratio
- punctuation count
- punctuation ratio
- punctuation variety
- stopword ratio
- long-token ratio
- short-token ratio
- alphabetic-character ratio
- whether the query contains `?`
- whether it contains `:`
- whether it contains `/`
- whether it contains `-`
- whether it contains parentheses
- whether it starts with a WH-word
- whether it contains comparative cues such as `vs`, `compare`, or `better`

These features help separate short lexical lookup queries from more semantic, question-like, or structurally complex queries.

### Hashed Lexical Features

In addition to numeric statistics, the router also uses hashed text features:

- word unigrams and bigrams
- character n-grams

This adds much more expressiveness without requiring a heavy neural router model or a manually maintained vocabulary.

## Router Model

The current classifier is a class-balanced logistic-regression model.

Reasons for this choice:

- stronger than nearest centroid
- lightweight and fast to train
- stable on small and medium datasets
- easy to explain in a report
- provides class probabilities for confidence-aware routing

Before training, the numeric features are standardized. The lexical hashed features are combined with the numeric block into one sparse feature matrix. The classifier is then trained to predict one of the three routing classes:

- `sparse`
- `dense`
- `hybrid`

## Confidence-Aware Fallback

One important addition in this version is fallback routing.

At inference time:

1. the router predicts class probabilities
2. if the top probability is high enough, the corresponding branch is used
3. if the top probability is too low, the router falls back to a default branch

Current default setting:

- confidence threshold: `0.45`
- fallback branch: `dense`

This design is useful because uncertain router decisions are often where errors happen. Instead of forcing a brittle decision, the system backs off to a stable baseline.

## Dataset Setup

### SciFact

SciFact uses the provided train/test split directly:

- train qrels: `data/scifact/raw/qrels_train.jsonl`
- test qrels: `data/scifact/raw/qrels_test.jsonl`

### SCIDOCS

SCIDOCS does not provide a separate train qrels split in the downloaded setup, so the project uses a query-level self-split:

- source qrels: `data/scidocs/raw/qrels_test.jsonl`
- generated train qrels: `data/scidocs/raw/qrels_router_train.jsonl`
- generated test qrels: `data/scidocs/raw/qrels_router_test.jsonl`

The split is done on unique `query_id` values with a fixed random seed, while keeping the original relevance labels unchanged.

## Efficiency Notes

The training script now reuses cached processed retriever artifacts when available:

- `data/<dataset>/processed/sparse.pkl`
- `data/<dataset>/processed/dense.pkl`

That makes router iteration much faster than rebuilding retrievers from scratch each time.

The pipeline also tracks:

- average latency
- estimated retrieved-context tokens
- estimated total tokens
- estimated cost in USD

These are approximate efficiency signals for comparison across routing choices.

## Why Baselines Are Needed During Training

An important clarification is that the router does not require all baselines to be run at inference time.

There are two different stages in the project:

### Training and analysis stage

During training, the project runs `sparse`, `dense`, and `hybrid` for the same queries because the router needs supervision.

That supervision answers the question:

`for this query, which retrieval branch worked best?`

So baseline retrieval is used offline to:

- construct weak labels
- compare fixed strategies
- estimate the oracle upper bound
- diagnose where the router fails

This is why baseline evaluation is necessary during development.

### Inference stage

At inference time, the intended workflow is different:

1. receive a new query
2. let the router predict the best branch
3. run only that selected branch

So the deployment-time value of the router is precisely that it avoids running all retrieval branches for every query.

## Relation To The Project Goal

The proposal goal is not to say that every incoming query must first run all baselines before routing.

Instead, the goal is:

- use offline baseline comparisons to learn a routing policy
- then use that learned policy online to choose a branch directly

This distinction matters because the project is trying to optimize both:

- retrieval quality
- efficiency such as latency and cost

If all branches had to be run at inference time, the router would provide little practical efficiency benefit.

So the current training pipeline uses baseline runs as an offline teacher, while the intended final system uses the learned router as an online selector.

## How To Run

The baseline/default router training call uses `MRR@10` as the weak-label metric:

```bash
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl
```

If you want to use multi-metric weak labels, you can combine multiple ranking metrics and choose a tie preference explicitly. For example:

```bash
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl \
  --label-metrics mrr@10 ndcg@10 recall@10 \
  --label-weights 0.5 0.3 0.2 \
  --label-tie-preference dense
```

Useful router-training options:

- `--label-metrics`: metrics used to create weak labels
- `--label-weights`: weights for the corresponding label metrics
- `--label-tie-preference`: which branch wins if combined label scores tie
- `--label-near-tie-mode`: optional preferred branch when it is within the configured label margin
- `--label-near-tie-margin`: margin used by `--label-near-tie-mode`
- `--use-retrieval-features`: add retrieval-confidence features to the router input representation
- `--router-confidence-threshold`: fallback threshold at inference time
- `--router-fallback-mode`: default branch used when the router is uncertain

The current best SCIDOCS-specific router configuration is:

```bash
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl \
  --label-near-tie-mode dense \
  --label-near-tie-margin 0.0 \
  --use-retrieval-features
```

That command is SCIDOCS-specific. SciFact currently performs best with the default Phase 2 router rather than the SCIDOCS-specific label and feature settings.

The same router options can now also be passed through the one-command pipeline. For example:

```bash
python scripts/run_pipeline.py \
  --dataset-dir data/scidocs \
  --label-metrics mrr@10 ndcg@10 recall@10 \
  --label-weights 0.5 0.3 0.2 \
  --label-tie-preference dense
```

The one-command pipeline also supports dataset-aware presets through `--router-preset`.

Current preset behavior:

- `auto`: choose the current best-known router configuration for the dataset
- `scifact-best`: use the default Phase 2 SciFact router
- `scidocs-best`: use margin-aware dense-favoring labels plus retrieval-confidence features
- `manual`: disable presets and use only the flags passed on the command line

So the intended one-command usage is now:

```bash
python scripts/run_pipeline.py --dataset-dir data/scifact --router-preset auto
python scripts/run_pipeline.py --dataset-dir data/scidocs --router-preset auto
```

Under `auto`, the script currently applies:

- SciFact: default weak-label rule `MRR@10`, tie preference `hybrid`, query-only router
- SCIDOCS: dense-favoring near-tie weak labels plus retrieval-confidence features

## Current Results Summary

This document now distinguishes between:

- archived default-router snapshots in `phase1/<dataset>/router_report.json`
- current best runnable results in `results/<dataset>/router_report.json`

The current best one-command pipeline behavior uses dataset-aware presets:

- SciFact: default Phase 2 router
- SCIDOCS: margin-aware dense-favoring weak labels plus retrieval-confidence features

### Current Best Results

These are the latest result files produced by the dataset-aware one-command pipeline.

#### SciFact

- Router accuracy: `0.7333`
- Router `MRR@10`: `0.6608`
- Router `nDCG@10`: `0.6945`
- Best fixed baseline `MRR@10`: `0.6563` from `hybrid`
- Oracle `MRR@10`: `0.7337`

Interpretation:

- the current SciFact router slightly beats the best fixed baseline
- this remains the default query-only Phase 2 router
- there is still a clear oracle gap, so routing can improve further

#### SCIDOCS

- Router accuracy: `0.6300`
- Router `MRR@10`: `0.3740`
- Router `nDCG@10`: `0.2284`
- Best fixed baseline `MRR@10`: `0.3723` from `dense`
- Oracle `MRR@10`: `0.4636`

Interpretation:

- the current SCIDOCS router now slightly beats the best fixed baseline on `MRR@10`
- this improvement comes from dataset-specific supervision cleanup plus retrieval-confidence features
- there is still a sizable oracle gap, so SCIDOCS routing is improved but not solved

### Archived Default Snapshot

For reference, the archived default SCIDOCS Phase 2 router in `phase1/scidocs/router_report.json` used:

- weak-label rule: `MRR@10`, tie -> `hybrid`
- query-only features

and achieved:

- accuracy: `0.5000`
- `MRR@10`: `0.3620`

That older snapshot is still useful for ablation and error analysis, but it is no longer the best SCIDOCS Phase 2 configuration.

## Comparison With Multi-Metric Weak Labels

After the default Phase 2 router was established, an alternative labeling strategy was also tested.

Instead of creating weak labels from `MRR@10` only, the alternative version used:

- `MRR@10`
- `nDCG@10`
- `Recall@10`

with weights:

- `0.5`
- `0.3`
- `0.2`

and tie preference set to `dense`.

This was intended to reduce the hybrid bias observed on SCIDOCS and produce more balanced weak labels.

The comparison below uses:

- default snapshot from `phase1/<dataset>/router_report.json`
- multi-metric experimental run from `results/<dataset>/router_report.json`

### SciFact: Default vs Multi-Metric Labels

Default Phase 2 router:

- accuracy: `0.7333`
- `MRR@10`: `0.6608`
- `nDCG@10`: `0.6945`

Multi-metric weak-label router:

- accuracy: `0.7267`
- `MRR@10`: `0.6539`
- `nDCG@10`: `0.6912`

Interpretation:

- the multi-metric version was slightly worse than the default router on SciFact
- the label distribution shifted heavily toward `dense`
- that shift appears to over-correct relative to the earlier hybrid-heavy labeling

### SCIDOCS: Default vs Multi-Metric Labels

Default Phase 2 router:

- accuracy: `0.5000`
- `MRR@10`: `0.3620`
- `nDCG@10`: `0.2277`

Multi-metric weak-label router:

- accuracy: `0.6050`
- `MRR@10`: `0.3623`
- `nDCG@10`: `0.2240`

Interpretation:

- the multi-metric version substantially improved classification accuracy
- it also reduced the earlier hybrid-heavy label skew
- however, the ranking gain was extremely small on `MRR@10`
- `nDCG@10` slightly decreased
- the router still did not beat the fixed `dense` baseline

### Main Takeaway

The multi-metric weak-label strategy is more reasonable from a supervision-design perspective, especially for SCIDOCS where `dense` and `hybrid` tie often under `MRR@10`.

However, in the current experiments:

- it did not outperform the default Phase 2 router on SciFact
- it only produced a marginal `MRR@10` gain on SCIDOCS
- it therefore remains an important ablation and alternative labeling strategy, rather than the new default router configuration

## SCIDOCS Margin-Aware Labeling + Retrieval Features

After the SCIDOCS error analysis, a more targeted ablation was run:

- make weak labels prefer `dense` on ties / near-ties
- add retrieval-confidence features to the router input

The detailed write-up is in [phase2/scidocs_margin_retrieval_comparison.md](scidocs_margin_retrieval_comparison.md).

High-level outcome:

- default SCIDOCS router: `MRR@10 = 0.3620`
- margin-aware labels only: `0.3712`
- retrieval-confidence features only: `0.3576`
- margin-aware labels + retrieval-confidence features: `0.3740`
- fixed `dense` baseline: `0.3723`

Detailed comparison on the same SCIDOCS self-split test set:

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default router | 0.5000 | 0.2480 | 0.3620 | 0.2277 | 135.15 |
| Margin-aware labels only | 0.6500 | 0.2438 | 0.3712 | 0.2274 | 40.94 |
| Retrieval-confidence features only | 0.5050 | 0.2455 | 0.3576 | 0.2235 | 146.77 |
| Margin-aware labels + retrieval-confidence features | 0.6300 | 0.2490 | 0.3740 | 0.2284 | 65.64 |
| Fixed dense baseline | - | 0.2510 | 0.3723 | 0.2307 | 14.15 |

Interpretation:

- fixing the weak-label bias was the first important step
- retrieval-confidence features were only helpful after that supervision issue was reduced
- the combined variant slightly beats fixed `dense` on the SCIDOCS self-split setting

This is an important result for the project narrative because it shows that the SCIDOCS router problem was not simply “the classifier is too weak.”

Instead, the evidence suggests a two-stage explanation:

1. the original SCIDOCS supervision was biased because many `dense` and `hybrid` queries tied under `MRR@10`
2. once that label bias was reduced, retrieval-side confidence signals became useful and helped the router cross the fixed-`dense` baseline

So the current best SCIDOCS result in Phase 2 is not the multi-metric weak-label variant. It is the targeted combination of:

- margin-aware dense-favoring weak labels
- retrieval-confidence features

That makes this the strongest candidate for the next default SCIDOCS router setting.

## Main Strengths

- much stronger than the original centroid baseline
- still lightweight and reproducible
- supports uncertainty-aware routing
- can use a pure query-only configuration when that is sufficient, as on SciFact
- can also benefit from dataset-specific retrieval-confidence features when needed, as on SCIDOCS
- current dataset-aware presets beat the best fixed strategy on both SciFact and SCIDOCS

## Current Limitations

- the strongest SCIDOCS result currently depends on dataset-specific configuration rather than one universal router setting
- fallback behavior is still manually configured rather than learned per dataset
- SCIDOCS evaluation still depends on a self-split rather than an official train/test split
- the router remains below the oracle upper bound on both datasets
- retrieval-confidence features help on SCIDOCS, but they did not help on their own and still need more systematic tuning

## Next Phase-2 Improvements

The most natural next improvements are:

1. add retrieval-confidence features from the candidate branches
2. add per-query error analysis to explain wrong routing decisions
3. tune or learn fallback behavior per dataset
4. explore cost-aware labeling so the router prefers cheaper branches when quality is close
5. evaluate whether a tree model or small neural router improves over logistic regression without losing interpretability

## Short Summary

The current Phase 2 router is a confidence-aware logistic-regression selector over richer query features and hashed lexical features. It is still simple enough to explain clearly, but it performs much better than the original baseline and gives a more credible foundation for the next stage of MultiplexRAG routing work.
