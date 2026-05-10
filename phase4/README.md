# Phase 4 Router Supervision And Decision Experiments

## Goal

Phase 4 focuses on the next router question after Phase 3:

`how can the router improve retrieval quality more reliably through better supervision and better decision rules?`

The main purpose of this phase is not to add more features blindly. Instead, the goal is to test a cleaner and more reportable improvement pipeline built around:

- weak-label design
- targeted retrieval-side features
- fallback and decision calibration
- quality-cost-aware routing

This phase should therefore be organized so that each experiment answers one specific question and its artifacts are easy to compare later in `REPORT.md`.

## Directory Layout

Phase 4 uses a dataset-first layout, with each dataset containing four subdirectories:

```text
phase4/
  README.md
  tables/
  scidocs/
    main/
    ablations/
    diagnostics/
  nfcorpus/
    main/
    ablations/
    diagnostics/
  scifact/
    main/
    ablations/
    diagnostics/
  fiqa/
    main/
    ablations/
    diagnostics/
```

### Meaning Of Each Subdirectory

- `main/`: the primary headline experiments for that dataset
- `ablations/`: controlled comparisons that isolate one design choice at a time
- `diagnostics/`: confusion-style summaries, label-distribution analyses, and error analysis outputs
- `tables/`: cross-dataset CSV or Markdown tables prepared for the final report

## Naming Rules

All Phase 4 artifacts should use short, compositional names that encode the experiment recipe without becoming too hard to read.

### File Prefix

Every trained router artifact should start with:

`router_`

### Recommended Name Pattern

Use this pattern for router artifacts:

`router_<label>_<features>_<decision>`

where:

- `<label>` describes the weak-label policy
- `<features>` describes the enabled retrieval feature family
- `<decision>` describes fallback or utility logic

### Label Tokens

Recommended label tokens:

- `base`: current baseline label rule
- `margin`: dense-favoring near-tie labeling
- `marginfilt`: margin-aware labeling with ambiguous-query filtering
- `multimetric`: combined label metrics
- `utilitylabel`: utility-aware label construction

### Feature Tokens

Recommended feature tokens:

- `qonly`: query-only router
- `basic`: basic retrieval-confidence features
- `basicqm`: basic + query_match
- `basicscore`: basic + score_shape
- `fullret`: larger retrieval-feature block

### Decision Tokens

Recommended decision tokens:

- `densefb`: dense fallback
- `hybridfb`: hybrid fallback
- `t045`: confidence threshold 0.45
- `t050`: confidence threshold 0.50
- `calib`: calibrated confidence rule
- `utility`: utility-aware decision rule

### Full Examples

- `router_margin_qonly_densefb_report.json`
- `router_margin_basic_densefb_report.json`
- `router_margin_basicqm_densefb_t045_report.json`
- `router_marginfilt_basicqm_densefb_t045_report.json`
- `router_margin_basicqm_hybridfb_t050_report.json`
- `router_utilitylabel_basicqm_utility_report.json`

This naming style keeps the files sortable while still making the experiment recipe readable from the filename.

## Artifact Types

Each router experiment should usually produce three primary files:

- `*.pkl`: trained router model
- `*_report.json`: summary metrics and metadata
- `*_predictions.jsonl`: per-query predictions

Recommended pairing:

- `router_margin_basicqm_densefb_t045.pkl`
- `router_margin_basicqm_densefb_t045_report.json`
- `router_margin_basicqm_densefb_t045_predictions.jsonl`

If an experiment also produces custom analysis outputs, place them under `diagnostics/` and reuse the same stem:

- `router_margin_basicqm_densefb_t045_confusion.json`
- `router_margin_basicqm_densefb_t045_error_slices.md`

## What Goes In `main/`

The `main/` directory should contain only the few experiments that are strong enough to appear in the final narrative.

Recommended Phase 4 `main/` candidates:

- current best reproduction
- margin-aware labels
- margin-aware + basic retrieval features
- margin-aware + basic + query_match
- threshold-tuned or calibrated variant
- utility-aware variant

This directory should stay small and high-signal.

## What Goes In `ablations/`

The `ablations/` directory should contain controlled comparisons such as:

- `qonly` vs `basic`
- `basic` vs `basicqm`
- `margin` vs `marginfilt`
- `densefb` vs `hybridfb`
- `t045` vs `t050`

These experiments are important, but they should not clutter the top-level dataset folder.

## What Goes In `diagnostics/`

The `diagnostics/` directory should contain artifacts that explain *why* a router variant helped or failed, for example:

- label distributions
- predicted label distributions
- confusion-style summaries
- dense-to-hybrid and hybrid-to-dense error counts
- representative failure examples
- short Markdown notes for dataset-specific findings

Suggested diagnostic filename stems:

- `diagnostics_label_distribution.json`
- `diagnostics_confusion.json`
- `diagnostics_error_examples.md`

or, when tied to one experiment:

- `router_margin_basicqm_densefb_t045_confusion.json`
- `router_margin_basicqm_densefb_t045_error_examples.md`

## Recommended Dataset Startup Order

To keep Phase 4 manageable, run datasets in this order:

1. `scidocs`
2. `nfcorpus`
3. `scifact`
4. `fiqa` only if needed as a negative-result check

This order matches the current project evidence:

- `scidocs` is the clearest supervision-noise case
- `nfcorpus` is the clearest fallback-policy case
- `scifact` is useful as a stability check
- `fiqa` is mainly useful to verify that a router should not be adopted automatically

## Recommended First Main Experiments

For a new dataset in Phase 4, start with these files in `main/`:

- `router_base_qonly_<fallback>_report.json`
- `router_margin_qonly_<fallback>_report.json`
- `router_margin_basic_<fallback>_report.json`
- `router_margin_basicqm_<fallback>_t045_report.json`

Then add one decision-focused experiment:

- `router_margin_basicqm_<fallback>_t050_report.json`
or
- `router_margin_basicqm_<fallback>_calib_report.json`

## Report-Friendly Table Files

Store final compact tables in `phase4/tables/`.

Recommended files:

- `phase4/tables/phase4_main_results.md`
- `phase4/tables/phase4_main_results.csv`
- `phase4/tables/phase4_ablation_results.md`
- `phase4/tables/phase4_error_summary.md`

This keeps report-ready material separate from raw model artifacts.
