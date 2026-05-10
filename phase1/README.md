# Phase 1 Summary

This folder stores the current Phase 1 comparison snapshot for the MultiplexRAG project.

## Scope

Phase 1 compares five decision policies:

- `sparse`
- `dense`
- `hybrid`
- `router`
- `oracle`

All main comparisons below use `top-k = 10`.

## Dataset Snapshots

### SciFact

This is the current local rerun produced from the restored `sentence-transformers` dense backend.

- Corpus: `data/scifact/raw/corpus.jsonl`
- Queries: `data/scifact/raw/queries.jsonl`
- Train qrels: `data/scifact/raw/qrels_train.jsonl`
- Test qrels: `data/scifact/raw/qrels_test.jsonl`
- Router report: [phase1/scifact/router_report.json](scifact/router_report.json)

| Method | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 0.7757 | 0.6184 | 0.6523 | 38.92 |
| Dense | 0.7900 | 0.6055 | 0.6479 | 6.09 |
| Hybrid | 0.8306 | 0.6563 | 0.6939 | 48.07 |
| Router | 0.8107 | 0.6608 | 0.6945 | 42.92 |
| Oracle | 0.8662 | 0.7337 | 0.7606 | 40.35 |

Router snapshot:

- Train queries: `809`
- Test queries: `300`
- Train label distribution: `dense=99`, `hybrid=639`, `sparse=71`
- Weak-label rule: `MRR@10`, tie -> `hybrid`
- Router classification accuracy: `0.7333`
- Router fallback setting: confidence `< 0.45 -> dense`

Takeaways:

- `hybrid` is the strongest fixed strategy on SciFact.
- the upgraded `router` now slightly beats the fixed `hybrid` baseline on `MRR@10` and `nDCG@10`.
- this gain comes from a stronger query representation and confidence-aware fallback, though dense remains the fastest fixed option.
- `oracle` is still clearly better, so there is room to improve the routing policy.

### SCIDOCS

This is the current second-dataset snapshot used to replace the earlier FiQA-based Phase 1 narrative.

- Corpus: `data/scidocs/raw/corpus.jsonl`
- Queries: `data/scidocs/raw/queries.jsonl`
- Official qrels: `data/scidocs/raw/qrels_test.jsonl`
- Router train qrels: `data/scidocs/raw/qrels_router_train.jsonl`
- Router test qrels: `data/scidocs/raw/qrels_router_test.jsonl`
- Router split metadata: `data/scidocs/raw/qrels_router_split_meta.json`
- Router report: [phase1/scidocs/router_report.json](scidocs/router_report.json)

Fixed-baseline evaluation on the full official qrels split:

| Method | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 0.1543 | 0.2680 | 0.1495 | 204.12 |
| Dense | 0.2307 | 0.3607 | 0.2166 | 72.97 |
| Hybrid | 0.2099 | 0.3343 | 0.1985 | 252.69 |

Router snapshot on the query-level self-split test set:

| Method | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 0.1660 | 0.3020 | 0.1669 | 214.16 |
| Dense | 0.2510 | 0.3723 | 0.2307 | 14.15 |
| Hybrid | 0.2428 | 0.3432 | 0.2178 | 206.96 |
| Router | 0.2480 | 0.3620 | 0.2277 | 135.15 |
| Oracle | 0.2635 | 0.4636 | 0.2600 | 140.03 |

Router snapshot:

- Router self-split test queries: `200`
- Weak-label rule: `MRR@10`, tie -> `hybrid`
- Router classification accuracy: `0.5000`
- Router fallback setting: confidence `< 0.45 -> dense`

Takeaways:

- `dense` is the strongest fixed strategy on SCIDOCS.
- This differs from SciFact, where `hybrid` is strongest.
- the upgraded `router` improves over the earlier centroid baseline, but still does not beat always-`dense` on SCIDOCS.
- `oracle` again shows that query-aware selection still has headroom.

## Cross-Dataset Interpretation

- The best fixed strategy is dataset-dependent: `hybrid` wins on SciFact, while `dense` wins on SCIDOCS.
- This directly supports the main proposal motivation that a single retrieval representation is not universally optimal.
- The upgraded router is now strong enough to beat the best fixed baseline on SciFact, but not yet on SCIDOCS.
- The clearest next step is to add retrieval-confidence signals and dataset-aware fallback logic.

## Notes On Splits And Cost

- SciFact uses the provided dataset split directly.
- SCIDOCS does not provide a separate train qrels split in the downloaded setup, so router training uses a query-level self-split derived from the official qrels with a fixed random seed.
- The Phase 1 snapshot above refers to the default router labeling rule: `MRR@10` with tie preference `hybrid`.
- Estimated token/cost fields are useful for relative comparison, but they should still be treated as approximate context-cost summaries rather than final billing-grade measurements.

## Files

- [phase1_results.json](phase1_results.json): structured multi-dataset Phase 1 snapshot
- [router_method.md](router_method.md): explanation of the current router design
- [phase1/scifact/router_report.json](scifact/router_report.json): current SciFact report
- [phase1/scidocs/router_report.json](scidocs/router_report.json): current SCIDOCS report
