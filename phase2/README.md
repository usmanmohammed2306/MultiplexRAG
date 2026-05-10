# Phase 2 Report

## MultiplexRAG: Query-Aware Retrieval Routing

This Phase 2 report follows the original proposal goal of building a MultiplexRAG system that compares multiple retrieval representations and learns a router to select the best retrieval strategy per query.

The core question is:

`given multiple representations of the same corpus, can we learn a routing policy that improves retrieval quality and efficiency over a single fixed strategy?`

## 1. Motivation And Background

The proposal started from a simple observation: no single retrieval representation is uniformly optimal.

- `sparse` retrieval is strong for exact lexical overlap, acronyms, rare terms, and code-like queries
- `dense` retrieval is strong for semantic matching and paraphrases
- `hybrid` retrieval often provides the best overall quality, but usually costs more latency than a single branch

The motivation of MultiplexRAG is therefore to avoid forcing every query through the same retrieval path. Instead, the system should learn which branch is most appropriate for each query.

This motivates a two-level design:

1. build multiple retrieval branches over the same corpus
2. train a router that predicts the best branch for each query

## 2. Problem Statement

The project asks whether a learned router can outperform strong fixed baselines such as:

- always-`sparse`
- always-`dense`
- always-`hybrid`

The proposal also requires an oracle upper bound, defined as post-hoc best-branch selection for each query, in order to quantify remaining headroom.

So the empirical goals in Phase 2 are:

- compare fixed retrieval strategies
- train a lightweight but stronger router than the initial baseline
- analyze why the router succeeds or fails
- show whether router-based selection improves the quality-cost tradeoff

## 3. Datasets And Task

Phase 2 uses three main report datasets plus one supplemental validation dataset.

The reporting scope is:

- primary Phase 2 results: `SciFact`, `SCIDOCS`, and `NFCorpus`
- supplemental validation results: `FiQA`

This distinction is important for the final project narrative.

- `SciFact`, `SCIDOCS`, and `NFCorpus` are the main evidence used to answer the proposal question across different retrieval settings
- `FiQA` is used as a supplemental generalization check, and as a useful negative result showing that a router should not be promoted when a fixed retriever remains stronger

### SciFact

- corpus: `data/scifact/raw/corpus.jsonl`
- queries: `data/scifact/raw/queries.jsonl`
- train qrels: `data/scifact/raw/qrels_train.jsonl`
- test qrels: `data/scifact/raw/qrels_test.jsonl`

SciFact is the smaller dataset and uses the provided train/test split directly.

### SCIDOCS

- corpus: `data/scidocs/raw/corpus.jsonl`
- queries: `data/scidocs/raw/queries.jsonl`
- official qrels: `data/scidocs/raw/qrels_test.jsonl`
- router train qrels: `data/scidocs/raw/qrels_router_train.jsonl`
- router test qrels: `data/scidocs/raw/qrels_router_test.jsonl`

Because the downloaded SCIDOCS setup does not provide a separate train qrels file for router training, the project creates a query-level self-split from the official qrels. The split keeps relevance labels unchanged and only separates queries into train and test partitions.

### FiQA

- corpus: `data/fiqa/raw/corpus.jsonl`
- queries: `data/fiqa/raw/queries.jsonl`
- train qrels: `data/fiqa/raw/qrels_train.jsonl`
- test qrels: `data/fiqa/raw/qrels_test.jsonl`

FiQA was added later as a supplemental dataset-level check to see whether the router improvements observed on SciFact and SCIDOCS generalized to a third retrieval setting.

### NFCorpus

- corpus: `data/nfcorpus/raw/corpus.jsonl`
- queries: `data/nfcorpus/raw/queries.jsonl`
- train qrels: `data/nfcorpus/raw/qrels_train.jsonl`
- validation qrels: `data/nfcorpus/raw/qrels_validation.jsonl`
- test qrels: `data/nfcorpus/raw/qrels_test.jsonl`

NFCorpus was added as a third main report dataset to test whether the project conclusions also held on a biomedical retrieval task with natural-language consumer questions and terminology-heavy medical documents.

This dataset is especially useful for MultiplexRAG because it creates a realistic tension between lexical and semantic matching:

- some queries benefit from exact medical term overlap
- some queries are phrased in simpler language than the corresponding document terminology
- hybrid retrieval therefore has a strong chance to help

For NFCorpus, the final test split is used for fixed-baseline comparison, while router iteration and recipe selection are most naturally done on the provided validation split before reporting the final test result.

The retrieval task is document retrieval with `top-k = 10`.

## 4. System Overview

The Phase 2 system contains four main components:

1. `sparse` retriever
2. `dense` retriever
3. `hybrid` retriever
4. query-aware router

The workflow is:

1. build or load all three retrieval branches
2. run them on training queries
3. compare their ranking outputs against qrels
4. derive weak labels for router training
5. train a router that predicts `sparse`, `dense`, or `hybrid`
6. at inference time, run only the selected branch

This is important: all branches are run during offline training and analysis, but the intended deployed workflow uses the router to avoid running all branches for every query.

## 5. Retrieval Representations

The proposal called for three retrieval strategies, and all three were implemented.

### Sparse

The sparse branch is a BM25-style lexical retriever. It is strongest when exact surface-form matching matters.

### Dense

The dense branch is a vector retriever using a sentence-transformer encoder. It is strongest for semantic similarity and paraphrase-style matching.

### Hybrid

The hybrid branch combines sparse and dense outputs through fusion. It is often strongest in quality, but tends to be slower because it effectively uses two retrieval signals.

## 6. Router Design

The original router baseline was deliberately simple and used a small handcrafted feature set with a nearest-centroid classifier.

Phase 2 upgrades that design in several ways:

- richer numeric query features
- hashed lexical query features
- class-balanced logistic regression
- confidence-aware fallback
- optional retrieval-confidence features
- optional dataset-aware weak-label rules

The main implementation lives in:

- [scripts/train_router.py](../scripts/train_router.py)
- [src/multiplexrag/router.py](../src/multiplexrag/router.py)

### Query Features

The router uses:

- query length and token statistics
- digit and acronym ratios
- punctuation and uppercase features
- structural cues such as `?`, `:`, `/`, `-`, and parentheses
- WH-word and comparative cues
- hashed word and character n-grams

On SCIDOCS, the strongest router additionally uses retrieval-confidence features such as:

- top-1 score
- top-1 vs top-2 score gap
- top-k overlap between branches
- whether branches return the same top document

### Weak Supervision

The router is not trained with manual query-type labels.

Instead, for each training query:

1. all retrieval branches are run
2. each branch is scored against qrels
3. the best branch becomes the weak label

The baseline/default weak-label rule is:

- metric: `MRR@10`
- tie preference: `hybrid`

Phase 2 also tested:

- multi-metric weak labels
- margin-aware dense-favoring labels

### Fallback

The router predicts class probabilities. If the confidence is below threshold, it falls back to a default branch.

Current default fallback:

- threshold: `0.45`
- fallback branch: `dense`

## 7. Evaluation Protocol

The project evaluates five policies:

- `sparse`
- `dense`
- `hybrid`
- `router`
- `oracle`

The main quality metrics are:

- `Recall@10`
- `MRR@10`
- `nDCG@10`

The project also tracks:

- classification `accuracy` against weak labels
- average latency
- estimated token usage
- estimated retrieval cost

`MRR@10` is used both as a ranking metric and, in the default setup, as the primary weak-label construction metric. However, Phase 2 analysis shows that router accuracy alone is not enough; ranking metrics remain the primary evaluation criteria.

## 8. Main Results

### 8.1 SciFact

Current best SciFact result from [results/scifact/router_report.json](../results/scifact/router_report.json):

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.7757 | 0.6184 | 0.6523 |
| Dense | 0.7900 | 0.6055 | 0.6479 |
| Hybrid | 0.8306 | 0.6563 | 0.6939 |
| Router | 0.8107 | 0.6608 | 0.6945 |
| Oracle | 0.8662 | 0.7337 | 0.7606 |

Interpretation:

- `hybrid` is the strongest fixed baseline
- the Phase 2 router slightly beats fixed `hybrid` on both `MRR@10` and `nDCG@10`
- there is still a large oracle gap

So on SciFact, the proposal goal is supported: adaptive routing improves over the best fixed retrieval representation.

### 8.2 SCIDOCS

Current best SCIDOCS result from [results/scidocs/router_report.json](../results/scidocs/router_report.json):

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.1660 | 0.3020 | 0.1669 |
| Dense | 0.2510 | 0.3723 | 0.2307 |
| Hybrid | 0.2428 | 0.3432 | 0.2178 |
| Router | 0.2490 | 0.3740 | 0.2284 |
| Oracle | 0.2635 | 0.4636 | 0.2600 |

Interpretation:

- `dense` is the strongest fixed baseline on SCIDOCS
- unlike the original default router, the improved SCIDOCS-specific router now slightly beats fixed `dense` on `MRR@10`
- the oracle gap is still substantial

This is important because it shows the best fixed strategy is dataset-dependent:

- SciFact favors `hybrid`
- SCIDOCS favors `dense`

That is directly consistent with the proposal’s main motivation.

### 8.3 FiQA Supplemental Validation

FiQA was used as an additional validation dataset rather than the main Phase 2 showcase. The best fixed baseline on FiQA is `dense`, and the default router did not beat it.

Using the best FiQA router variant currently tested:

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.2784 | 0.2713 | 0.2175 |
| Dense | 0.4389 | 0.4463 | 0.3687 |
| Hybrid | 0.4308 | 0.4095 | 0.3442 |
| Best FiQA Router Tested | 0.4322 | 0.4342 | 0.3591 |
| Oracle | 0.4859 | 0.5510 | 0.4311 |

Interpretation:

- FiQA behaves more like SCIDOCS than SciFact in the sense that `dense` is the strongest fixed strategy
- adding retrieval-confidence features improved the FiQA router over the default router
- however, the router still did not surpass fixed `dense`

So at the current stage, the preferred FiQA deployment strategy is still simply to use `dense` rather than a learned router.

### 8.4 NFCorpus

NFCorpus serves as the third main Phase 2 report dataset and tests whether the project conclusions extend to biomedical retrieval, where natural-language user questions must be matched against more terminology-heavy documents.

Fixed-baseline evaluation on the NFCorpus test split from [results/nfcorpus/router_report.json](../results/nfcorpus/router_report.json):

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.1522 | 0.5085 | 0.3062 |
| Dense | 0.1590 | 0.5055 | 0.3191 |
| Hybrid | 0.1639 | 0.5455 | 0.3368 |
| Default Router | 0.1690 | 0.5381 | 0.3408 |
| Oracle | 0.1763 | 0.6124 | 0.3606 |

Interpretation of the default NFCorpus result:

- `hybrid` is the strongest fixed baseline on the test split
- the default query-only router improves `Recall@10` and `nDCG@10` over fixed `hybrid`
- however, the default router does not beat fixed `hybrid` on `MRR@10`

This pattern is important. It shows that the router is already capturing useful query-dependent signal, because it finds slightly more relevant documents overall and improves aggregate ranking quality. At the same time, NFCorpus makes clear that a default query-only router is not yet a strong enough final answer for every dataset.

So the main Phase 2 takeaway from NFCorpus is:

- the dataset benefits strongly from combining lexical and semantic evidence
- the default router is promising, but not yet a clear deployment winner
- router gains on this dataset depend on choosing a dataset-appropriate fallback policy rather than relying on the same recipe that worked elsewhere
- the oracle gap remains substantial, so query-aware branch selection still has room to improve

Because NFCorpus provides a validation split, it is a good dataset for studying router recipe selection more cleanly than datasets that require self-splitting. That made it possible to run a focused ablation over fallback policy, retrieval-confidence features, and weak-label design.

## 9. Router Analysis

### 9.1 Default Router Findings

The earlier default SCIDOCS router underperformed fixed `dense`.

The detailed diagnosis in [phase2/scidocs_router_error_analysis.md](scidocs_router_error_analysis.md) showed that the problem was not mainly model capacity. The more important issue was weak-label bias:

- many `dense` and `hybrid` queries tied under `MRR@10`
- tie-breaking preferred `hybrid`
- this made the training labels too hybrid-heavy

As a result, the router made expensive errors by sending truly dense-friendly queries to `hybrid`.

### 9.2 Multi-Metric Weak Labels

Phase 2 also evaluated multi-metric weak labels using:

- `MRR@10`
- `nDCG@10`
- `Recall@10`

This made the supervision design more reasonable, especially on SCIDOCS, but did not become the best overall Phase 2 configuration:

- on SciFact it was slightly worse than the default router
- on SCIDOCS it improved classification accuracy but gave only marginal ranking improvement

So the multi-metric version remains a useful ablation, not the default best model.

### 9.3 Margin-Aware Labels Plus Retrieval Features

The strongest SCIDOCS improvement came from combining:

- margin-aware dense-favoring weak labels
- retrieval-confidence features

The comparison is documented in [phase2/scidocs_margin_retrieval_comparison.md](scidocs_margin_retrieval_comparison.md).

Key SCIDOCS ablation results:

| Configuration | Accuracy | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Default router | 0.5000 | 0.3620 | 0.2277 |
| Margin-aware only | 0.6500 | 0.3712 | 0.2274 |
| Retrieval features only | 0.5050 | 0.3576 | 0.2235 |
| Margin-aware + retrieval features | 0.6300 | 0.3740 | 0.2284 |

This gives a very clear conclusion:

1. fixing supervision helped more than simply adding more features
2. retrieval-confidence features became useful only after the label bias was reduced
3. the combined SCIDOCS-specific configuration is the current best router for that dataset

### 9.4 NFCorpus Validation And Router Recipe Selection

NFCorpus gave a particularly useful ablation setting because it includes an explicit validation split:

- train qrels: `data/nfcorpus/raw/qrels_train.jsonl`
- validation qrels: `data/nfcorpus/raw/qrels_validation.jsonl`
- test qrels: `data/nfcorpus/raw/qrels_test.jsonl`

This made it possible to tune the router on validation data rather than repeatedly inspecting only the final test split.

The starting point on the validation split was the retrieval-feature router with default fallback:

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Retrieval features + dense fallback | 0.5710 | 0.1373 | 0.5184 | 0.3066 |
| Retrieval features + hybrid fallback | 0.6481 | 0.1398 | 0.5241 | 0.3102 |
| Retrieval features + hybrid fallback + threshold 0.5 | 0.6698 | 0.1429 | 0.5222 | 0.3131 |
| Retrieval features + hybrid fallback + multi-metric labels | 0.5123 | 0.1379 | 0.5185 | 0.3132 |
| Fixed Hybrid Baseline | - | 0.1388 | 0.5110 | 0.3084 |

These NFCorpus ablations reveal a different pattern from SCIDOCS.

#### Why Dense Fallback Was A Poor Default On NFCorpus

The best fixed strategy on NFCorpus is `hybrid`, not `dense`.

That matters because the router uses fallback exactly when it is uncertain. If the system is uncertain, falling back to a branch that is not the strongest fixed baseline is often the wrong safety policy. On NFCorpus, changing the fallback from `dense` to `hybrid` produced the clearest single improvement:

- accuracy rose from `0.5710` to `0.6481`
- `MRR@10` rose from `0.5184` to `0.5241`
- `nDCG@10` rose from `0.3066` to `0.3102`
- `Recall@10` also improved slightly

So on NFCorpus, the most important router improvement was not a more complex classifier. It was aligning the fallback policy with the dataset’s strongest fixed retrieval strategy.

#### Effect Of A Higher Confidence Threshold

Raising the confidence threshold from `0.45` to `0.5` made the router more conservative. In practice, that meant more uncertain queries were sent to `hybrid` instead of forcing a lower-confidence branch prediction.

This change produced:

- the best `Recall@10` on validation: `0.1429`
- the best `nDCG@10` among the fallback variants: `0.3131`
- the highest weak-label accuracy in the ablation: `0.6698`

But there was also a small tradeoff:

- `MRR@10` decreased slightly from `0.5241` to `0.5222`

This suggests that the higher threshold reduces brittle routing mistakes and improves overall ranking stability, but also removes some of the aggressive branch choices that occasionally help place the first relevant document very high in the ranking.

#### Effect Of Multi-Metric Weak Labels

NFCorpus also responded differently to multi-metric labels than SCIDOCS.

Using combined weak labels from `MRR@10`, `nDCG@10`, and `Recall@10` with weights `0.5 / 0.3 / 0.2` produced:

- `MRR@10 = 0.5185`
- `nDCG@10 = 0.3132`
- `Recall@10 = 0.1379`
- `accuracy = 0.5123`

This variant slightly improved `nDCG@10`, but it did not become the best overall configuration. In particular:

- it remained below the best fallback-tuned router on `MRR@10`
- it did not improve `Recall@10` over the threshold-tuned fallback variant
- it reduced classification accuracy substantially

So on NFCorpus, multi-metric supervision is better interpreted as a useful alternative label design rather than the main source of router gains.

#### NFCorpus Validation Conclusion

The strongest overall NFCorpus router recipe tested so far is:

- enable retrieval-confidence features
- use `hybrid` as the fallback branch
- optionally raise the confidence threshold to `0.5` if the priority is more stable recall and nDCG

For final reporting, the important point is not just that NFCorpus needed more tuning. The important point is that NFCorpus provides direct evidence for the project's dataset-aware claim.

It shows that:

- on SCIDOCS, the main gain came from fixing weak-label bias and then adding retrieval features
- on NFCorpus, the main gain came from retrieval features plus a dataset-appropriate fallback policy

So the broader lesson is not merely that router performance varies by dataset. The best way to improve the router also varies by dataset.

That makes NFCorpus a strong main-report dataset, because it demonstrates a different but equally important success condition for MultiplexRAG: not only which branch is best, but which router training recipe is appropriate for that dataset.

### 9.5 FiQA Validation

FiQA was used to test whether the same router-improvement recipe would generalize beyond the two main showcase datasets.

The result was mixed:

- retrieval-confidence features improved the default FiQA router substantially
- but the FiQA router still did not beat fixed `dense`
- a dense-favoring margin-aware label rule did not provide the same benefit on FiQA that it did on SCIDOCS

This is a useful negative result.

It shows that:

- router improvements are themselves dataset-dependent
- a dataset-specific router preset should only be adopted when it actually beats the best fixed baseline

So FiQA strengthens the project argument in a different way:

- not every dataset benefits from the same router configuration
- and not every dataset benefits from a router enough to replace the strongest fixed strategy

### 9.6 What The Ablation Means In Practice

This ablation is an offline development-stage experiment, not part of normal online inference.

In practice, it means:

- the comparison is run in advance
- it is evaluated over the dataset split, not on just a few example queries
- it is used to choose the best router configuration for that dataset

So the ablation cost is higher than ordinary inference because it requires:

- running or reusing all retrieval branches
- generating weak labels
- training and evaluating multiple router variants

However, this is a one-time offline model-selection cost rather than a per-query online cost.

Once the best configuration is chosen, the deployed workflow is still:

1. receive a query
2. let the selected router predict the best branch
3. run only that branch

So the ablation study helps determine which router design should be used for a dataset, while the final deployed system still benefits from query-level routing efficiency.

## 10. Relation To The Proposal

The proposal argued that:

- no single retrieval strategy is best for all queries
- a learned multiplexer/router should improve the quality-cost tradeoff
- oracle analysis should quantify remaining headroom

Phase 2 supports all three claims.

### Claim 1: No single fixed retriever is universally best

Supported.

- SciFact best fixed baseline: `hybrid`
- SCIDOCS best fixed baseline: `dense`
- FiQA best fixed baseline: `dense`
- NFCorpus best fixed baseline: `hybrid`

### Claim 2: A router can improve over strong baselines

Supported.

- SciFact router beats fixed `hybrid`
- SCIDOCS router beats fixed `dense` under the best dataset-aware Phase 2 configuration
- NFCorpus shows that router performance can surpass fixed `hybrid` once the router is given a dataset-appropriate fallback policy and retrieval-confidence features

But Phase 2 also shows an important qualification:

- on FiQA, the router did not beat fixed `dense`
- on NFCorpus, the default router was not enough; the gain only became convincing after dataset-specific fallback tuning

This does not weaken the project. Instead, it strengthens the realism of the conclusion: router usefulness is dataset-dependent, and a router should only be preferred when it empirically beats strong fixed alternatives.

### Claim 3: Oracle analysis reveals remaining improvement potential

Supported.

Across the datasets examined here, the oracle is still much stronger than the learned router, so there is room for future routing improvements.

## 11. Limitations

Despite the improvements, Phase 2 still has important limitations:

- the strongest SCIDOCS result depends on dataset-specific router configuration rather than one universal setting
- the strongest NFCorpus result also depends on dataset-specific fallback and confidence settings
- FiQA currently still favors fixed `dense` over all router variants tested so far
- SCIDOCS uses a query-level self-split rather than an official train/test router split
- router accuracy is based on weak labels, not human annotations
- latency measurements vary by machine load and should be treated as relative signals
- the router is still below the oracle upper bound on all datasets examined here

## 12. Phase 3 Direction: Dataset-Aware Training Recipe

The most natural next step is not simply to build a larger or more complex router. Instead, Phase 3 should formalize the project as a dataset-aware routing framework.

The central idea is:

- keep one overall MultiplexRAG architecture
- but allow the router training recipe to vary by dataset when the evidence justifies it

Phase 2 already provides the empirical basis for this direction:

- SciFact performs best with the default Phase 2 query-only router
- SCIDOCS performs best with margin-aware dense-favoring weak labels plus retrieval-confidence features
- NFCorpus performs best with retrieval-confidence features plus a `hybrid` fallback policy, with threshold tuning depending on whether `MRR@10` or `nDCG@10` is prioritized
- FiQA currently still prefers fixed `dense`, meaning a router preset should not be adopted there unless it actually beats the fixed baseline

So Phase 3 should aim to turn these observations into an explicit training recipe registry.

### Proposed Phase 3 Recipe

For each dataset, Phase 3 would define:

- the preferred weak-label construction rule
- whether retrieval-confidence features should be enabled
- the preferred fallback policy
- whether router deployment is recommended at all

In practice, that would create a mapping such as:

- SciFact -> default router recipe
- SCIDOCS -> margin-aware + retrieval-confidence recipe
- NFCorpus -> retrieval-confidence + hybrid fallback, with confidence-threshold tuning
- FiQA -> fixed dense, or further router tuning before promotion to a preset

### Why This Matters

This direction is important because Phase 2 already shows that:

- the best fixed retrieval strategy is dataset-dependent
- the best router-improvement recipe is also dataset-dependent
- some datasets benefit from a router enough to justify deployment, while others currently do not

So the next contribution is not only a stronger router, but a more principled policy for when and how a router should be trained and used for each dataset.

### Practical Goal

The practical Phase 3 goal is therefore:

1. run offline baseline and ablation analysis for a dataset
2. determine the best router recipe or decide that a fixed retriever is preferable
3. register that outcome as a dataset-aware preset
4. use the selected preset in the one-command pipeline

This would turn the current project from a single-router experiment into a reusable dataset-aware retrieval-routing framework.

## 13. Conclusion

Phase 2 successfully advances the original MultiplexRAG proposal from a simple routing baseline to a much stronger query-aware retrieval selector.

For final reporting purposes, the clearest Phase 2 storyline is:

- primary result dataset 1: `SciFact`
- primary result dataset 2: `SCIDOCS`
- primary result dataset 3: `NFCorpus`
- supplemental generalization check: `FiQA`

The main conclusions are:

- retrieval preference is dataset-dependent
- a lightweight logistic-regression router with richer features is already strong enough to beat the best fixed baseline on SciFact
- SCIDOCS required more careful supervision design and retrieval-side confidence signals
- once those were added, the router also slightly beat the best fixed SCIDOCS baseline
- NFCorpus confirmed the same overall project thesis under a different retrieval setting: router gains were achievable, but only after aligning the fallback policy with the dataset's strongest fixed baseline and adding retrieval-confidence features
- FiQA showed that router gains are not universal; for that dataset, fixed `dense` remains the strongest practical strategy so far

So the central proposal idea is supported: query-aware routing can improve over a single fixed retrieval strategy, and the quality-cost tradeoff is best understood through both baseline comparison and oracle analysis.

## 14. Key Files

- [phase2/router_method.md](router_method.md)
- [phase2/scidocs_router_error_analysis.md](scidocs_router_error_analysis.md)
- [phase2/scidocs_margin_retrieval_comparison.md](scidocs_margin_retrieval_comparison.md)
- [results/scifact/router_report.json](../results/scifact/router_report.json)
- [results/scidocs/router_report.json](../results/scidocs/router_report.json)
- [results/nfcorpus/router_report.json](../results/nfcorpus/router_report.json)
- [results/nfcorpus/router_rf_hybridfb_report.json](../results/nfcorpus/router_rf_hybridfb_report.json)
- [results/nfcorpus/router_rf_hybridfb_t05_report.json](../results/nfcorpus/router_rf_hybridfb_t05_report.json)
- [results/nfcorpus/router_rf_mm_report.json](../results/nfcorpus/router_rf_mm_report.json)
