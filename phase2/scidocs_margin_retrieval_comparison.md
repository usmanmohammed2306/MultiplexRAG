# SCIDOCS Margin-Aware Router Comparison

## Goal

This comparison tests two targeted fixes for the SCIDOCS router:

1. make weak labels margin-aware by preferring `dense` on ties / near-ties
2. add retrieval-confidence features to the router input

The motivation comes from the earlier SCIDOCS error analysis:

- default weak labels were biased toward `hybrid`
- many `dense` and `hybrid` queries tied under `MRR@10`
- the main ranking loss came from misrouting truly dense-friendly queries away from `dense`

## Configurations Compared

All numbers below use the same SCIDOCS router self-split test set.

### 1. Default Phase 1 router

- weak-label rule: `MRR@10`, tie -> `hybrid`
- retrieval-confidence features: off

### 2. Margin-aware labels only

- weak-label rule: `MRR@10`
- if `dense` is tied with the best label score, choose `dense`
- retrieval-confidence features: off

### 3. Retrieval-confidence features only

- weak-label rule: `MRR@10`, tie -> `hybrid`
- retrieval-confidence features: on

### 4. Margin-aware labels + retrieval-confidence features

- weak-label rule: `MRR@10`
- if `dense` is tied with the best label score, choose `dense`
- retrieval-confidence features: on

## Results

Fixed dense baseline on the same self-split test set:

- `MRR@10`: `0.3723`
- `nDCG@10`: `0.2307`

Router variants:

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default router | 0.5000 | 0.2480 | 0.3620 | 0.2277 | 135.15 |
| Margin-aware only | 0.6500 | 0.2438 | 0.3712 | 0.2274 | 40.94 |
| Retrieval features only | 0.5050 | 0.2455 | 0.3576 | 0.2235 | 146.77 |
| Margin-aware + retrieval features | 0.6300 | 0.2490 | 0.3740 | 0.2284 | 65.64 |

## Main Findings

### Margin-aware weak labels helped a lot

Changing the label rule alone produced the largest single gain:

- `MRR@10`: `0.3620 -> 0.3712`
- accuracy: `0.5000 -> 0.6500`

This almost closed the gap to fixed `dense`.

That supports the earlier diagnosis that SCIDOCS was suffering from weak-label bias more than from pure model weakness.

### Retrieval-confidence features alone did not help

When retrieval-confidence features were added without fixing the weak labels:

- `MRR@10`: `0.3620 -> 0.3576`
- `nDCG@10`: `0.2277 -> 0.2235`

So better features could not overcome noisy supervision by themselves.

### The combination was best

The strongest SCIDOCS result in this comparison came from combining both fixes:

- router `MRR@10`: `0.3740`
- fixed `dense` `MRR@10`: `0.3723`

This means the updated router now slightly beats the best fixed baseline on this SCIDOCS self-split setting.

It also improves over the default router on:

- `MRR@10`
- `nDCG@10`
- classification accuracy

while keeping latency well below the original hybrid-heavy router behavior.

This combined configuration is now the basis for the current SCIDOCS dataset-aware preset used by the one-command pipeline.

## Interpretation

The comparison suggests a clear ordering of what mattered most:

1. fix the supervision
2. then add richer routing signals

In other words:

- the label rule was the primary bottleneck
- retrieval-confidence features became useful only after the label noise was reduced

## Artifacts

Relevant experiment files:

- [results/scidocs_ablation/router_margin_0.0.json](../results/scidocs_ablation/router_margin_0.0.json)
- [results/scidocs_ablation/retrieval_only_cached.summary.json](../results/scidocs_ablation/retrieval_only_cached.summary.json)
- [results/scidocs_ablation/margin_dense_retrieval_cached.summary.json](../results/scidocs_ablation/margin_dense_retrieval_cached.summary.json)
- [results/scidocs/router_report.json](../results/scidocs/router_report.json)
- [phase1/scidocs/router_report.json](../phase1/scidocs/router_report.json)
