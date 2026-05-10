# Single-Dataset Guide

This project already supports running one dataset at a time.

The key idea is simple:

- each dataset lives under `data/<dataset_name>/`
- indexes are built per dataset
- retrieval outputs are written to `results/<dataset_name>/`
- the main entry point is `--dataset-dir`

## Supported Local Datasets

The repository currently contains these dataset directories:

- `data/scifact`
- `data/scidocs`
- `data/fiqa`

## When To Use Which Dataset

### `scifact`

Use this when you want the smallest and cleanest end-to-end baseline.

- smallest main dataset in the project
- has train/test qrels directly
- best fixed baseline in current docs: `hybrid`
- recommended when you want faster iteration

### `scidocs`

Use this when you want the main larger comparison dataset from the later project phases.

- larger than `scifact`
- router uses a generated query-level train/test split
- best fixed baseline in current docs: `dense`
- best router recipe is dataset-specific

### `fiqa`

Use this when you want an additional validation dataset.

- included locally
- behaves more like `scidocs` than `scifact`
- current docs say fixed `dense` is the strongest practical choice
- useful as a third-dataset check, not the main showcase dataset

## Fastest Way To Run One Dataset

If the dataset is already prepared under `data/<name>/raw`, run:

```bash
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/scifact
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/scidocs
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/fiqa
```

This runs:

1. index building
2. sparse retrieval
3. dense retrieval
4. hybrid retrieval
5. evaluation
6. router training and router evaluation

## If You Only Want Baselines

If you want to evaluate one dataset without training the router:

```bash
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/scifact --skip-router
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/scidocs --skip-router
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/fiqa --skip-router
```

This is especially reasonable for `fiqa`, because the current project conclusion is that fixed `dense` is still better than the tested router variants.

## If You Want Just One Retrieval Mode

You can skip the full pipeline and run only one retrieval branch:

```bash
./.venv/bin/python scripts/build_index.py --dataset-dir data/scifact
./.venv/bin/python scripts/retrieve.py --dataset-dir data/scifact --mode dense --topk 10 --out results/scifact/dense.jsonl
./.venv/bin/python scripts/evaluate.py --dataset-dir data/scifact --pred results/scifact/dense.jsonl
```

Replace `data/scifact` with `data/scidocs` or `data/fiqa` as needed.

Available retrieval modes:

- `sparse`
- `dense`
- `hybrid`

## Recommended Single-Dataset Setups

### Option A: Smallest End-To-End Demo

```bash
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/scifact
```

Choose this if you want the cleanest full MultiplexRAG demonstration.

### Option B: Main Report Dataset

```bash
./.venv/bin/python scripts/run_pipeline.py --dataset-dir data/scidocs
```

Choose this if you want to align with the stronger Phase 2 narrative.

### Option C: Fixed Dense Validation

```bash
./.venv/bin/python scripts/build_index.py --dataset-dir data/fiqa
./.venv/bin/python scripts/retrieve.py --dataset-dir data/fiqa --mode dense --topk 10 --out results/fiqa/dense.jsonl
./.venv/bin/python scripts/evaluate.py --dataset-dir data/fiqa --pred results/fiqa/dense.jsonl
```

Choose this if you want to use `fiqa` in the way most consistent with the current project findings.

## Preparing A New Dataset Without Mixing It With Others

If you want to add another BEIR dataset and keep it isolated:

```bash
./.venv/bin/python scripts/prepare_data.py --dataset BeIR/nfcorpus --qrels-dataset BeIR/nfcorpus-qrels
./.venv/bin/python scripts/build_index.py --dataset-dir data/nfcorpus
```

That creates a separate dataset folder and does not affect the existing ones.

## Practical Recommendation

If your goal is to "use individual datasets" for the class project, the cleanest choices are:

- use only `scifact` if you want a compact and stable demo
- use only `scidocs` if you want the stronger main experimental story
- use `scifact + scidocs` if you want to show dataset-dependent retrieval behavior
- use `fiqa` only as an extra validation dataset, not the main result
