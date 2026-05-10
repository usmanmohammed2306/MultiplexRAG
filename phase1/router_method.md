# Current Router Method

## Goal

The router does not retrieve documents directly. Its job is to choose one retrieval strategy for each query:

- `sparse`
- `dense`
- `hybrid`

The overall learning target is:

`query text -> best retrieval strategy`

## Where The Process Lives

The current router pipeline is implemented across these files:

- [scripts/train_router.py](../scripts/train_router.py): end-to-end routing experiment
- [src/multiplexrag/router.py](../src/multiplexrag/router.py): query features, oracle labeling rule, and confidence-aware logistic router
- [src/multiplexrag/retrieval.py](../src/multiplexrag/retrieval.py): sparse, dense, and hybrid retrievers used to generate router supervision
- [src/multiplexrag/eval_utils.py](../src/multiplexrag/eval_utils.py): ranking metrics used during labeling and evaluation

## End-to-End Process

The current routing process is:

1. Load corpus, queries, and qrels.
2. Build or load the three retrieval strategies: `sparse`, `dense`, and `hybrid`.
3. Run all three retrievers for every query.
4. Compare their per-query ranking quality against qrels.
5. Assign each training query a weak label equal to the best-performing strategy.
6. Convert each query into numeric query features plus hashed lexical features.
7. Train a class-balanced logistic-regression router with confidence-aware fallback.
8. Use the trained classifier to predict the best strategy for test queries.
9. Evaluate the router against fixed baselines and the oracle upper bound.

So the router is not a separate retrieval model. It is a query-to-strategy selector that sits on top of the three retrievers.

## How Supervision Is Created

The router is trained with weak supervision derived from retrieval outcomes rather than manual query-type labels.

For each training query:

1. Run all three retrievers.
2. Compare the resulting ranked lists against qrels using a weak-label scoring rule.
3. Assign the query the label of the best-performing strategy.

This means the router is learning a strategy-selection policy, not document relevance itself.

In code, the weak labels are produced by:

- collecting query results for all three modes in [scripts/train_router.py](../scripts/train_router.py)
- selecting the best mode with `best_mode_for_query(...)` in [src/multiplexrag/router.py](../src/multiplexrag/router.py)

Current default weak-label rule:

- metric: `MRR@10`
- tie preference: `hybrid`

Optional router-training settings now also support:

- multi-metric weak labels such as `MRR@10 + nDCG@10 + Recall@10`
- custom label weights
- custom tie preference such as `dense`

## Current Query Features

The current implementation now uses a richer query-only feature set from [router.py](../src/multiplexrag/router.py):

- character length
- token length
- average token length
- unique-token ratio
- number of digits
- digit ratio
- number of acronyms
- acronym ratio
- uppercase ratio
- punctuation count and diversity
- stopword ratio
- long-token and short-token ratios
- whether the query contains a question mark
- whether it contains structural cues such as `:`, `/`, `-`, or parentheses
- whether it starts with a WH-word
- whether it contains comparative cues such as `vs`, `compare`, or `better`

In addition to those numeric features, the router also uses hashed lexical features:

- word unigrams and bigrams
- character n-grams

This keeps the router query-only and reproducible, while giving it substantially more expressive power than the original 7-feature baseline.

## Current Model

The current policy is a confidence-aware logistic-regression classifier.

Training:

1. Convert each training query into numeric query statistics plus hashed word/character features.
2. Fit a class-balanced multinomial logistic-regression model over weak labels: `dense`, `sparse`, `hybrid`.
3. Keep a configurable confidence threshold and fallback mode.

Inference:

1. Convert a test query into the same feature representation.
2. Predict class probabilities for `dense`, `sparse`, and `hybrid`.
3. If the top probability is below the confidence threshold, fall back to the configured default strategy.
4. Otherwise choose the highest-probability strategy.

The current default configuration is:

- weak-label metrics: `MRR@10`
- weak-label tie preference: `hybrid`
- confidence threshold: `0.45`
- fallback mode: `dense`

So the router is now a lightweight probabilistic strategy selector rather than a nearest-centroid baseline.

## Datasets And Splits

### SciFact

SciFact uses the provided dataset split directly:

- training qrels from `data/scifact/raw/qrels_train.jsonl`
- test qrels from `data/scifact/raw/qrels_test.jsonl`

### SCIDOCS

SCIDOCS does not provide a separate training qrels split in the downloaded setup. For router experiments, we therefore create a query-level self-split from the official qrels:

- source qrels: `data/scidocs/raw/qrels_test.jsonl`
- generated train qrels: `data/scidocs/raw/qrels_router_train.jsonl`
- generated test qrels: `data/scidocs/raw/qrels_router_test.jsonl`
- split metadata: `data/scidocs/raw/qrels_router_split_meta.json`

Split protocol:

1. Keep the official relevance labels unchanged.
2. Split on unique `query_id`, not individual qrels rows.
3. Use a fixed random seed for reproducibility.
4. Use the generated split only for router training/evaluation.

## Cost Tracking

The current pipeline also tracks estimated context cost:

- estimated query tokens
- estimated retrieved-context tokens
- estimated total tokens
- estimated cost in USD using a configurable per-1K-token price

These are useful for relative quality-cost comparisons, but they should still be interpreted as approximate retrieval-to-generation cost estimates rather than final API billing numbers.

## Why This Is Useful

- fast enough for iterative experiments
- easy to explain in a proposal or report
- suitable as a first routing baseline across multiple datasets
- exposes a meaningful oracle gap for future work
- supports configurable confidence-aware fallback without rerunning retrieval design

## Current Limitations

- the router still relies only on query-side features
- it does not yet use retrieval-confidence features such as top-score gaps or rank-list overlap
- fallback behavior is global rather than dataset-adaptive
- SCIDOCS router evaluation uses a self-split rather than an official train/test split
- the router improves over the earlier centroid baseline, but still does not beat the strongest fixed strategy on SCIDOCS

## Recommended Next Improvements

- add retrieval-confidence features
- tune the fallback policy per dataset or learn it directly
- explore cost-aware labeling so the router prefers cheaper strategies when quality is close
- add explicit per-query error analysis to explain where the router diverges from oracle behavior
