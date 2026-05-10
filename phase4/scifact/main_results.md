# Phase 4 SciFact Main Results

## Summary Table

| Setting | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) | Avg Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | - | 0.8306 | 0.6563 | 0.6939 | 23.19 | 2208.98 | 0.004418 |
| Router: base + qonly + densefb + t=0.45 | 0.7333 | 0.8107 | 0.6608 | 0.6945 | 42.30 | 2210.93 | 0.004422 |
| Router: base + basic + densefb + t=0.45 | 0.7333 | 0.8403 | 0.6529 | 0.6971 | 12.32 | 2203.36 | 0.004407 |
| Router: base + basic + query_match + densefb + t=0.45 | 0.7433 | 0.8303 | 0.6585 | 0.6982 | 19.05 | 2208.17 | 0.004416 |
| Router: base + basic + query_match + densefb + t=0.50 | 0.7100 | 0.8312 | 0.6559 | 0.6971 | 12.29 | 2209.28 | 0.004419 |

## Main Takeaways

- The best SciFact `MRR@10` result remained the original query-only router with `dense` fallback and threshold `0.45`.
- `basic + query_match` remained competitive on SciFact and improved `nDCG@10`, but it did not surpass the default router on `MRR@10`.
- Raising the confidence threshold from `0.45` to `0.50` did not help on SciFact.
- This makes SciFact closer to NFCorpus than SCIDOCS with respect to threshold behavior.

## Delta Vs Hybrid Baseline

| Setting | Delta Recall@10 | Delta MRR@10 | Delta nDCG@10 | Delta Latency (ms) | Delta Tokens | Delta Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Router: base + qonly + densefb + t=0.45 | -0.0199 | +0.0045 | +0.0006 | +19.12 | +1.95 | +0.000004 |
| Router: base + basic + densefb + t=0.45 | +0.0098 | -0.0034 | +0.0032 | -10.86 | -5.62 | -0.000011 |
| Router: base + basic + query_match + densefb + t=0.45 | -0.0002 | +0.0022 | +0.0043 | -4.14 | -0.81 | -0.000002 |
| Router: base + basic + query_match + densefb + t=0.50 | +0.0006 | -0.0004 | +0.0032 | -10.89 | +0.30 | +0.000001 |

## Interpretation

The Phase 4 SciFact results play a different role from the corresponding SCIDOCS and NFCorpus experiments. On SciFact, the goal was not mainly to find a dramatically better router, because the baseline query-only router was already strong. Instead, the goal was to test whether the later transferable feature recipe remained competitive on a dataset where the original lightweight router already worked well.

That is exactly what the results show. The `basic + query_match` recipe remained competitive, improving `nDCG@10` from `0.6945` to `0.6982` while keeping `MRR@10` close to the best router. However, unlike SCIDOCS and NFCorpus, SciFact did not benefit from pushing the router toward a more conservative threshold. This further strengthens the cross-dataset conclusion that the transferable part of the later router design is the targeted feature family, while the best calibration policy still depends on the dataset.
