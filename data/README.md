# Data Directory Guide

This directory stores dataset-specific inputs, indexes, and temporary download artifacts for the MultiplexRAG project.

## Active Datasets

The project currently treats these as the main experiment datasets:

- `data/scifact/`
- `data/scidocs/`

Each dataset follows the same layout:

```text
data/
  <dataset_name>/
    raw/
    processed/
    .tmp_beir/
```

## Folder Purposes

### `data/<dataset>/raw`

Stores the raw files used by indexing, retrieval, evaluation, and router training.

Typical files:

- `corpus.jsonl`
- `queries.jsonl`
- `qrels_train.jsonl`
- `qrels_test.jsonl`

Some datasets only provide a test qrels split. In that case, only the available qrels files will appear.

Router experiments may also create:

- `qrels_router_train.jsonl`
- `qrels_router_test.jsonl`
- `qrels_router_split_meta.json`

### `data/<dataset>/processed`

Stores derived artifacts built from the raw dataset.

Typical files:

- `corpus.pkl`
- `dense.pkl`
- `sparse.pkl`

These can always be regenerated from the corresponding `raw/` directory.

### `data/<dataset>/.tmp_beir`

Stores temporary download and extraction artifacts created while preparing BEIR datasets.

This is a staging/cache area rather than the canonical experiment location.

## Current Contents

### `data/scifact/`

SciFact is the small benchmark used as the compact baseline dataset.

- `data/scifact/raw/` stores corpus, queries, and train/test qrels
- `data/scifact/processed/` stores the built sparse and dense indexes

### `data/scidocs/`

SCIDOCS is the larger replacement dataset used for the second-stage comparison.

- `data/scidocs/raw/` stores corpus, queries, the official test qrels, and the generated router-only train/test split files
- `data/scidocs/processed/` stores generated index artifacts
- `data/scidocs/.tmp_beir/` stores the BEIR download cache for this dataset

## Dataset Notes

### SciFact

Recommended commands:

```bash
python scripts/build_index.py --dataset-dir data/scifact
python scripts/retrieve.py --dataset-dir data/scifact --mode hybrid --out results/scifact/hybrid.jsonl
python scripts/train_router.py \
  --dataset-dir data/scifact \
  --model-out results/scifact/router.pkl \
  --report-out results/scifact/router_report.json \
  --pred-out results/scifact/router_predictions.jsonl
```

### SCIDOCS

Recommended commands:

```bash
python scripts/build_index.py --dataset-dir data/scidocs
python scripts/retrieve.py --dataset-dir data/scidocs --mode hybrid --out results/scidocs/hybrid.jsonl
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl
```

If SCIDOCS only has `qrels_test.jsonl` in `raw/`, the pipeline generates:

- `data/scidocs/raw/qrels_router_train.jsonl`
- `data/scidocs/raw/qrels_router_test.jsonl`
- `data/scidocs/raw/qrels_router_split_meta.json`

## Adding Another Dataset

To add another BEIR dataset, create a new dataset-specific directory with:

```bash
python scripts/prepare_data.py --dataset BeIR/<name> --qrels-dataset BeIR/<name>-qrels --source beir-zip
python scripts/build_index.py --dataset-dir data/<name>
```

## Usage Recommendation

Prefer commands that use `--dataset-dir` so each experiment stays isolated:

```bash
python scripts/build_index.py --dataset-dir data/scifact
python scripts/build_index.py --dataset-dir data/scidocs
```

This keeps experiments reproducible and avoids dataset collisions.
