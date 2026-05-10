# Phase 4 SCIDOCS Ablation Results

## Best Main Configuration Reference

The Phase 4 main-stage reference model is:

- `margin + basic + query_match + dense fallback + t=0.45`

Reference metrics:

| Setting | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) | Avg Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main reference: `densefb + t=0.45` | 0.6250 | 0.2485 | 0.3771 | 0.2306 | 49.98 | 1650.40 | 0.003301 |

## Ablation Table

| Setting | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) | Avg Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main reference: `densefb + t=0.45` | 0.6250 | 0.2485 | 0.3771 | 0.2306 | 49.98 | 1650.40 | 0.003301 |
| Ablation: `densefb + t=0.50` | 0.6700 | 0.2500 | 0.3801 | 0.2309 | 64.92 | 1646.21 | 0.003292 |
| Ablation: `hybridfb + t=0.45` | 0.5900 | 0.2465 | 0.3761 | 0.2295 | 46.34 | 1656.54 | 0.003313 |

## Delta Vs Main Reference

| Setting | Delta Accuracy | Delta Recall@10 | Delta MRR@10 | Delta nDCG@10 | Delta Latency (ms) | Delta Tokens | Delta Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ablation: `densefb + t=0.50` | +0.0450 | +0.0015 | +0.0030 | +0.0003 | +14.94 | -4.20 | -0.000008 |
| Ablation: `hybridfb + t=0.45` | -0.0350 | -0.0020 | -0.0010 | -0.0012 | -3.64 | +6.14 | +0.000012 |

## Interpretation

These ablation results strengthen the Phase 4 SCIDOCS conclusion in two ways.

First, the strongest router recipe was not fragile to a small threshold adjustment. Raising the confidence threshold from `0.45` to `0.50` improved all four headline metrics that matter most in this experiment: accuracy, Recall@10, MRR@10, and nDCG@10. The resulting configuration produced the best SCIDOCS router result observed so far in Phase 4:

- `MRR@10 = 0.3801`
- `nDCG@10 = 0.2309`

This suggests that the best SCIDOCS router benefits from a slightly more conservative decision rule, where uncertain queries are more often routed back to `dense`.

Second, changing the fallback policy from `dense` to `hybrid` did not improve the best feature recipe. The `hybrid`-fallback variant remained competitive, but it was still worse than the `dense`-fallback version on Recall@10, MRR@10, nDCG@10, and accuracy. This supports the earlier SCIDOCS diagnosis that the most expensive router mistakes are the ones that send truly dense-friendly queries away from `dense`.

Taken together, these ablations show that the Phase 4 gains do not come only from better features. They also depend on choosing the right routing policy. On SCIDOCS, the current best policy is:

- margin-aware weak labels
- `basic + query_match` retrieval-side features
- `dense` fallback
- confidence threshold `0.50`
