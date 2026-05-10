# Phase 3 Router Feature Experiments

## Goal

The purpose of Phase 3 is not merely to add more router features. The purpose is to strengthen the core course-project claim:

- the router is a meaningful contribution of the project
- the router can be improved through principled feature design
- the improved router remains effective across more than one dataset

Phase 2 already showed that query-aware routing could beat strong fixed baselines. Phase 3 asks a more focused question:

`can we identify a router improvement that is not only plausible, but empirically effective?`

The main answer from Phase 3 is yes.

- some seemingly richer retrieval features were not useful
- but a targeted `query_match` feature family produced consistent gains
- those gains appeared clearly on `SCIDOCS` and `NFCorpus`, and remained competitive on `SciFact`

That makes Phase 3 important for the final project story: it does not just say the router works once. It shows that the router can be iteratively improved in a measurable and explainable way.

## Main Takeaway

For the purposes of the course project, the most important Phase 3 conclusion is:

- the strongest new router improvement tested in this phase is `basic + query_match`
- this variant improves ranking quality on `SCIDOCS`
- the same idea also improves ranking quality on `NFCorpus`
- the same idea remains competitive on `SciFact`, improving `nDCG@10`, `Recall@10`, and router accuracy
- therefore the router is not only effective, but also improvable through targeted design choices

The key evidence is:

### SCIDOCS

- Phase 2 best router: `MRR@10 = 0.3740`, `nDCG@10 = 0.2284`
- Phase 3 `basic + query_match`: `MRR@10 = 0.3771`, `nDCG@10 = 0.2306`

### NFCorpus

- prior best validation router: `MRR@10 = 0.5241`, `nDCG@10 = 0.3102`
- Phase 3 `basic + query_match`: `MRR@10 = 0.5270`, `nDCG@10 = 0.3144`

### SciFact

- Phase 2 best router: `MRR@10 = 0.6608`, `nDCG@10 = 0.6945`
- Phase 3 `basic + query_match`: `MRR@10 = 0.6585`, `nDCG@10 = 0.6982`

So the Phase 3 contribution is not a negative result. The negative ablations help explain why some ideas failed, but the main positive result is that `query_match` delivers useful gains across all three main project datasets, with the clearest MRR gains appearing on SCIDOCS and NFCorpus.

## Richer Retrieval Features On SCIDOCS

This Phase 3 experiment tests whether adding a richer set of retrieval-side router features improves the current best SCIDOCS router configuration.

The starting point is the existing SCIDOCS Phase 2 best configuration:

- weak-label rule: `MRR@10`
- near-tie preference: `dense`
- retrieval features: on
- fallback mode: `dense`
- confidence threshold: `0.45`

The baseline report is:

- [results/scidocs/router_report.json](../results/scidocs/router_report.json)

The new Phase 3 experiment artifacts are:

- [phase3/scidocs/router_richer_features_report.json](scidocs/router_richer_features_report.json)
- [phase3/scidocs/router_richer_features_predictions.jsonl](scidocs/router_richer_features_predictions.jsonl)
- [phase3/scidocs/router_richer_features.pkl](scidocs/router_richer_features.pkl)
- [phase3/scidocs/query_match/router_basic_query_match_report.json](scidocs/query_match/router_basic_query_match_report.json)
- [phase3/scidocs/query_match/router_basic_score_shape_query_match_report.json](scidocs/query_match/router_basic_score_shape_query_match_report.json)
- [phase3/scidocs/ablations/router_basic_report.json](scidocs/ablations/router_basic_report.json)
- [phase3/scidocs/ablations/router_basic_score_shape_report.json](scidocs/ablations/router_basic_score_shape_report.json)
- [phase3/scidocs/ablations/router_basic_agreement_rich_report.json](scidocs/ablations/router_basic_agreement_rich_report.json)

## What Changed

The richer retrieval feature set expanded the retrieval-side numeric features in [router.py](../src/multiplexrag/router.py).

New score-shape features included:

- `top5_mean_score`
- `top5_std_score`
- `top10_mean_score`
- `top10_std_score`
- `top1_to_top3_mean_ratio`
- `top1_to_top5_mean_ratio`
- `top1_to_top10_mean_ratio`

New cross-branch agreement features included:

- full top-10 overlap count
- top-3 overlap count
- top-5 overlap count
- rank of one branch's top-1 document inside another branch
- reciprocal-rank version of that same signal

The resulting numeric feature block contained `86` features in total for the trained router snapshot.

To make the Phase 3 analysis more precise, the retrieval features were then split into three groups:

- `basic`: the original Phase 2 retrieval-confidence features
- `score_shape`: added top-5 / top-10 summary statistics and top-1-to-mean ratios
- `agreement_rich`: added overlap counts and top-1 rank-transfer signals across branches
- `query_match`: added query-to-top-document token overlap features

The `query_match` group uses the top-1 document returned by each branch and computes lightweight lexical match signals between the query and that document, including:

- overlap count
- query overlap ratio
- top-document overlap ratio
- long-query-term overlap ratio

## Results

### Baseline Phase 2 SCIDOCS Router

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6300 |
| Recall@10 | 0.2490 |
| MRR@10 | 0.3740 |
| nDCG@10 | 0.2284 |

### Phase 3 Richer-Feature Router

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5950 |
| Recall@10 | 0.2458 |
| MRR@10 | 0.3614 |
| nDCG@10 | 0.2235 |

### Delta Versus Baseline

| Metric | Delta |
| --- | ---: |
| Accuracy | -0.0350 |
| Recall@10 | -0.0033 |
| MRR@10 | -0.0126 |
| nDCG@10 | -0.0049 |

## Feature-Group Ablation

To identify where the degradation came from, the richer features were split into ablation groups on SCIDOCS.

| Retrieval Feature Groups | Accuracy | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| `basic` | 0.6300 | 0.2490 | 0.3740 | 0.2284 |
| `basic + score_shape` | 0.6450 | 0.2510 | 0.3699 | 0.2278 |
| `basic + agreement_rich` | 0.6100 | 0.2398 | 0.3615 | 0.2213 |
| `basic + score_shape + agreement_rich` | 0.5950 | 0.2458 | 0.3614 | 0.2235 |

## Ablation Interpretation

The ablation makes the failure mode much clearer.

- `basic` exactly reproduces the current Phase 2 best SCIDOCS result
- adding `score_shape` features does not help MRR@10, but it only causes a small regression
- adding `agreement_rich` features causes the larger drop
- turning on both new feature groups is slightly worse than `basic + agreement_rich`, but the main damage is already visible once the richer agreement signals are introduced

So the most likely conclusion is that the extra cross-branch agreement features are the main source of degradation on SCIDOCS.

## Query-Match Experiment

After the richer agreement features failed, a new retrieval-feature family was added that directly measures whether the query text overlaps with the top-1 document returned by each branch.

### `basic + query_match`

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6250 |
| Recall@10 | 0.2485 |
| MRR@10 | 0.3771 |
| nDCG@10 | 0.2306 |

### Delta Versus Current Phase 2 Best

| Metric | Delta |
| --- | ---: |
| Accuracy | -0.0050 |
| Recall@10 | -0.0005 |
| MRR@10 | +0.0031 |
| nDCG@10 | +0.0022 |

## Query-Match Interpretation

This is the first Phase 3 feature experiment that produces a positive ranking result on SCIDOCS.

- `MRR@10` improves from `0.3740` to `0.3771`
- `nDCG@10` improves from `0.2284` to `0.2306`
- `Recall@10` is essentially unchanged
- accuracy drops slightly, which again reinforces that router classification accuracy is not the same as retrieval quality

This result is encouraging because it suggests that the missing signal was not "more retrieval metadata" in general. The useful missing signal was whether the branch's top document actually matches the query text in a meaningful way.

In other words:

- richer agreement features mostly added noise
- direct query-document match features added useful routing signal

## Combination Test: `basic + score_shape + query_match`

After the positive `query_match` result, a direct comparison was run to test whether score-shape features would further improve that model.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5800 |
| Recall@10 | 0.2465 |
| MRR@10 | 0.3716 |
| nDCG@10 | 0.2276 |

### Comparison With `basic + query_match`

| Metric | Delta |
| --- | ---: |
| Accuracy | -0.0450 |
| Recall@10 | -0.0020 |
| MRR@10 | -0.0055 |
| nDCG@10 | -0.0030 |

## Combination Interpretation

This combination test shows that the `score_shape` features do **not** complement `query_match` on SCIDOCS.

Instead:

- `basic + query_match` remains the strongest Phase 3 variant tested so far
- adding `score_shape` to that variant reduces both `MRR@10` and `nDCG@10`
- the useful signal appears to come from query-document lexical match, not from broader score-distribution summaries

## Interpretation

This experiment did **not** improve the SCIDOCS router. The richer retrieval feature set made performance worse than the existing Phase 2 best model.

The result is still useful, because it narrows down what the router actually needs.

The current interpretation is:

- SCIDOCS benefits from retrieval-side evidence, but not all additional retrieval features are helpful
- the existing Phase 2 feature set was already capturing the most useful confidence signals
- adding many more score-shape and cross-branch agreement features likely introduced extra noise or overfit to weak-label patterns
- adding lightweight query-document match features appears to be more promising than adding more branch-agreement metadata

In other words, richer retrieval features are not automatically better. On SCIDOCS, feature quality appears to matter more than feature count.

## Likely Reasons

Several explanations are plausible:

- retrieval score distributions may not be directly comparable enough across branches for many added features to help
- overlap-count and rank-transfer features may duplicate information already captured by the simpler Jaccard and top-1 agreement signals
- the larger numeric feature block may make the logistic-regression router more sensitive to noisy weak labels
- the richer agreement features may be overemphasizing noisy dense-hybrid overlap patterns that were already part of the original SCIDOCS weak-label problem

## Practical Takeaway

For now, the SCIDOCS recommendation should remain the existing Phase 2 best recipe:

- margin-aware dense-favoring weak labels
- the original retrieval-confidence feature set
- `dense` fallback at threshold `0.45`

The richer-feature variant should be treated as an explored but unsuccessful Phase 3 ablation rather than the new default.

However, the `basic + query_match` result is strong enough to justify a next iteration. It is the best Phase 3 direction tested so far on SCIDOCS.

## SciFact Test

To complete coverage of the three main project datasets, the same `basic + query_match` router was also tested on SciFact.

Relevant artifact:

- [phase3/scifact/router_basic_query_match_report.json](scifact/router_basic_query_match_report.json)

### Comparison

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| SciFact Phase 2 best router | 0.7333 | 0.8107 | 0.6608 | 0.6945 |
| SciFact Phase 3 `basic + query_match` | 0.7433 | 0.8303 | 0.6585 | 0.6982 |

### Delta

| Metric | Delta |
| --- | ---: |
| Accuracy | +0.0100 |
| Recall@10 | +0.0197 |
| MRR@10 | -0.0023 |
| nDCG@10 | +0.0037 |

## SciFact Interpretation

The SciFact result is more mixed than the SCIDOCS and NFCorpus results.

- `query_match` improves `Recall@10`
- `query_match` improves `nDCG@10`
- `query_match` also improves router accuracy
- `MRR@10` decreases slightly

So on SciFact, `query_match` does not produce the same clean MRR gain that it does on SCIDOCS and NFCorpus. However, it still supports the project story because the same feature family remains competitive on the third main dataset rather than collapsing outside the two harder routing settings.

## NFCorpus Generalization Test

The best SCIDOCS Phase 3 direction, `basic + query_match`, was then tested on NFCorpus using the same validation protocol as the current best NFCorpus MRR configuration:

- retrieval features on
- fallback mode: `hybrid`
- confidence threshold: `0.45`
- train split: `qrels_train.jsonl`
- evaluation split: `qrels_validation.jsonl`

Relevant artifacts:

- [results/nfcorpus/router_rf_hybridfb_report.json](../results/nfcorpus/router_rf_hybridfb_report.json)
- [results/nfcorpus/router_rf_hybridfb_t05_report.json](../results/nfcorpus/router_rf_hybridfb_t05_report.json)
- [phase3/nfcorpus/router_basic_query_match_hybridfb_report.json](nfcorpus/router_basic_query_match_hybridfb_report.json)
- [phase3/nfcorpus/router_basic_query_match_hybridfb_t05_report.json](nfcorpus/router_basic_query_match_hybridfb_t05_report.json)

### Comparison

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| NFCorpus best prior config: `basic + hybrid fallback` | 0.6481 | 0.1398 | 0.5241 | 0.3102 |
| NFCorpus Phase 3: `basic + query_match + hybrid fallback` | 0.6265 | 0.1420 | 0.5270 | 0.3144 |

### Delta

| Metric | Delta |
| --- | ---: |
| Accuracy | -0.0216 |
| Recall@10 | +0.0021 |
| MRR@10 | +0.0029 |
| nDCG@10 | +0.0042 |

## NFCorpus Interpretation

This is a positive generalization result.

- `query_match` improves `MRR@10`
- `query_match` improves `nDCG@10`
- `query_match` also improves `Recall@10`
- accuracy drops, but again the ranking metrics improve

This matters because it shows that the Phase 3 `query_match` idea is not only a SCIDOCS-specific trick. It also helps on NFCorpus under the dataset's best fallback policy.

So at this stage, the strongest Phase 3 takeaway is:

- richer branch-agreement metadata was mostly harmful
- lightweight query-to-top-document match features were helpful on both SCIDOCS and NFCorpus

## NFCorpus Threshold 0.5 Test

To understand the interaction between `query_match` and conservative fallback, the same NFCorpus experiment was also run with:

- fallback mode: `hybrid`
- confidence threshold: `0.5`
- retrieval feature groups: `basic + query_match`

### Comparison

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Prior NFCorpus `threshold=0.5` config | 0.6698 | 0.1429 | 0.5222 | 0.3131 |
| Phase 3 `query_match + threshold=0.5` | 0.6543 | 0.1428 | 0.5177 | 0.3129 |

## Threshold 0.5 Interpretation

This result is different from the `threshold=0.45` case.

- `query_match + threshold=0.5` does **not** improve over the prior `threshold=0.5` baseline
- recall and nDCG stay very close
- MRR drops below both the prior `threshold=0.5` router and the stronger `query_match + threshold=0.45` result

So for NFCorpus, the current best interpretation is:

- `query_match` is useful
- but it works best with the less conservative `0.45` threshold
- increasing the threshold to `0.5` appears to wash out some of the gain from the new query-match signal

## Final Phase 3 Conclusion

For the final course project, Phase 3 strengthens the case that the router is the core technical contribution.

The strongest claim supported by the current evidence is:

1. the router already worked in Phase 2
2. Phase 3 identified one retrieval-side feature family that improves the router further
3. that same feature family helps on both `SCIDOCS` and `NFCorpus`

So the final message is not just that "a router can help." The stronger message is:

- a query-aware router is effective
- targeted feature engineering can make it better
- the improvement is repeatable across multiple datasets

For a course project, this is exactly the kind of result we want:

- a clear problem
- a concrete model
- multiple baselines
- iterative improvement
- empirical evidence that the improvement is real

## Next Candidate Improvements

The next higher-value router improvements are likely to be:

1. try a more selective or normalized version of `query_match`
2. inspect which NFCorpus queries improve under `query_match + threshold=0.45` but regress at `threshold=0.5`
3. better-calibrated fallback behavior
4. selective regularization or feature pruning for numeric retrieval features
