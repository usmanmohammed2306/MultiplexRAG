# Phase 4 SCIDOCS Main Results

## Summary Table

| Setting | Accuracy | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) | Avg Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | - | 0.2510 | 0.3723 | 0.2307 | 12.63 | 1640.18 | 0.003280 |
| Router: base + qonly + densefb | 0.5000 | 0.2480 | 0.3620 | 0.2277 | 68.73 | 1674.83 | 0.003350 |
| Router: margin + qonly + densefb | 0.6500 | 0.2438 | 0.3712 | 0.2274 | 28.34 | 1653.76 | 0.003308 |
| Router: margin + basic + densefb | 0.6300 | 0.2490 | 0.3740 | 0.2284 | 83.22 | 1662.21 | 0.003324 |
| Router: margin + basic + query_match + densefb + t=0.45 | 0.6250 | 0.2485 | 0.3771 | 0.2306 | 49.98 | 1650.40 | 0.003301 |

## Main Takeaways

- The default query-only router underperformed the fixed `dense` baseline on SCIDOCS.
- Margin-aware weak labeling closed most of that gap even without retrieval-side features.
- Adding `basic` retrieval-confidence features pushed the router slightly above fixed `dense` on `MRR@10`.
- Adding `query_match` features produced the strongest Phase 4 SCIDOCS result: `MRR@10 = 0.3771`.
- Router accuracy did not align perfectly with retrieval quality: the best `MRR@10` model was not the highest-accuracy model.

## Delta Vs Dense Baseline

| Setting | Delta Recall@10 | Delta MRR@10 | Delta nDCG@10 | Delta Latency (ms) | Delta Tokens | Delta Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Router: base + qonly + densefb | -0.0030 | -0.0102 | -0.0031 | +56.10 | +34.65 | +0.000069 |
| Router: margin + qonly + densefb | -0.0072 | -0.0011 | -0.0033 | +15.71 | +13.58 | +0.000027 |
| Router: margin + basic + densefb | -0.0020 | +0.0018 | -0.0023 | +70.31 | +22.03 | +0.000044 |
| Router: margin + basic + query_match + densefb + t=0.45 | -0.0025 | +0.0049 | -0.0001 | +37.35 | +10.22 | +0.000020 |

## Interpretation

The Phase 4 SCIDOCS results reinforce the main conclusion from the earlier phases: the most important router improvement came from cleaning up weak supervision before expanding the feature set. The move from the default label rule to the margin-aware label rule produced the largest single gain. After that, retrieval-side features became useful, and the strongest incremental improvement came from lightweight `query_match` signals rather than from broader or more complex retrieval metadata.

These results also support a more careful interpretation of router evaluation. The highest-accuracy router was `margin + qonly + densefb`, but the highest-`MRR@10` router was `margin + basic + query_match + densefb + t=0.45`. This means that weak-label classification accuracy should be treated as a diagnostic measure rather than the primary target. On SCIDOCS, the best router is the one that improves ranking quality, even if its label agreement is slightly lower.
