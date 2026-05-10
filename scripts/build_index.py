#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Build dense/sparse indexes from corpus.")
    p.add_argument("--dataset-dir", default="", help="Dataset root containing raw/ and processed/ folders")
    p.add_argument("--corpus", default="data/raw/corpus.jsonl")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dense-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    from multiplexrag.data import load_corpus
    from multiplexrag.retrieval import DenseRetriever, SparseRetriever, save_pickle

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    corpus_path = dataset_dir / "raw" / "corpus.jsonl" if dataset_dir else Path(args.corpus)
    outdir = dataset_dir / "processed" if dataset_dir else Path(args.outdir)

    corpus = load_corpus(corpus_path)
    outdir.mkdir(parents=True, exist_ok=True)

    sparse = SparseRetriever().fit(corpus)
    dense = DenseRetriever(
        model_name=args.dense_model,
        dense_dim=args.dense_dim,
        batch_size=args.batch_size,
    ).fit(corpus)

    save_pickle(outdir / "sparse.pkl", sparse)
    save_pickle(outdir / "dense.pkl", dense)
    save_pickle(outdir / "corpus.pkl", corpus)

    print(f"Indexed {len(corpus)} documents")
    print(f"Corpus path  -> {corpus_path}")
    print(f"Sparse index -> {outdir / 'sparse.pkl'}")
    print(f"Dense index  -> {outdir / 'dense.pkl'}")
    print(f"Dense backend: {dense.backend}")


if __name__ == "__main__":
    main()
