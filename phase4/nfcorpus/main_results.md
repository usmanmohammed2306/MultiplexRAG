# Phase 4 NFCorpus Main Results

## Summary Table

| Setting | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) | Avg Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | - | 0.1388 | 0.5110 | 0.3084 | 7.95 | 2204.09 | 0.004408 |
| Router: base + qonly + hybridfb | 0.6728 | 0.1389 | 0.5203 | 0.3104 | 6.96 | 2204.61 | 0.004409 |
| Router: base + basic + hybridfb | 0.6481 | 0.1398 | 0.5241 | 0.3102 | 5.11 | 2215.80 | 0.004432 |
| Router: base + basic + query_match + hybridfb + t=0.45 | 0.6265 | 0.1420 | 0.5270 | 0.3144 | 6.54 | 2220.04 | 0.004440 |
| Router: base + basic + query_match + hybridfb + t=0.50 | 0.6543 | 0.1428 | 0.5177 | 0.3129 | 5.32 | 2212.09 | 0.004424 |

## Main Takeaways

- The best Phase 4 NFCorpus router remained `basic + query_match + hybrid fallback + t=0.45`.
- `query_match` features again improved the router over the `basic` retrieval-feature baseline.
- Increasing the confidence threshold from `0.45` to `0.50` hurt `MRR@10` on NFCorpus.
- This differs from SCIDOCS, where a higher threshold helped, and therefore strengthens the dataset-aware routing claim.

## Delta Vs Hybrid Baseline

| Setting | Delta Recall@10 | Delta MRR@10 | Delta nDCG@10 | Delta Latency (ms) | Delta Tokens | Delta Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Router: base + qonly + hybridfb | +0.0001 | +0.0093 | +0.0019 | -0.98 | +0.52 | +0.000001 |
| Router: base + basic + hybridfb | +0.0010 | +0.0131 | +0.0018 | -2.83 | +11.71 | +0.000024 |
| Router: base + basic + query_match + hybridfb + t=0.45 | +0.0032 | +0.0160 | +0.0059 | -1.40 | +15.95 | +0.000032 |
| Router: base + basic + query_match + hybridfb + t=0.50 | +0.0040 | +0.0068 | +0.0044 | -2.62 | +8.00 | +0.000016 |

## Interpretation

The Phase 4 NFCorpus results reinforce two parts of the project's overall argument.

First, they confirm that the `basic + query_match` feature recipe is not a SCIDOCS-only phenomenon. On NFCorpus, adding `query_match` to the `basic` retrieval-feature block again produced the strongest `MRR@10` result, improving the router from `0.5241` to `0.5270` and further widening the gap over the fixed `hybrid` baseline.

Second, the threshold-tuning result shows that decision calibration remains dataset-specific. On SCIDOCS, a higher threshold improved the router by making it more conservative. On NFCorpus, that same move reduced `MRR@10` from `0.5270` to `0.5177`. So the transferable part of the Phase 4 recipe is the targeted feature design, while the best fallback threshold still depends on the dataset.
