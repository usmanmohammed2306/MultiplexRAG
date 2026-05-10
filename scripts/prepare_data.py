#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download BeIR dataset parts and export to dataset-specific jsonl directories."
    )
    p.add_argument("--dataset", default="BeIR/scifact", help="HF dataset repo for corpus/queries")
    p.add_argument("--qrels-dataset", default="BeIR/scifact-qrels", help="HF dataset repo for qrels")
    p.add_argument("--name", default="", help="Local dataset folder name, defaults to dataset slug")
    p.add_argument("--outdir", default="", help="Output directory, defaults to data/<name>/raw")
    p.add_argument(
        "--source",
        choices=["auto", "hf", "beir-zip"],
        default="auto",
        help="Data source strategy: Hugging Face, BEIR zip, or auto fallback.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return p.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def fail_if_exists(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_slug(dataset_name: str) -> str:
    # "BeIR/scifact" -> "scifact"
    return dataset_name.split("/")[-1].replace("-qrels", "")


def export_corpus(dataset_name: str, outdir: Path, overwrite: bool):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, "corpus")
    split = ds["corpus"]
    out = outdir / "corpus.jsonl"
    ensure_parent(out)
    fail_if_exists(out, overwrite)

    def iter_rows():
        for item in split:
            yield {
                "doc_id": item.get("_id"),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }

    write_jsonl(out, iter_rows())
    return len(split), out


def export_queries(dataset_name: str, outdir: Path, overwrite: bool):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, "queries")
    split = ds["queries"]
    out = outdir / "queries.jsonl"
    ensure_parent(out)
    fail_if_exists(out, overwrite)

    def iter_rows():
        for item in split:
            yield {
                "query_id": item.get("_id"),
                "text": item.get("text", ""),
            }

    write_jsonl(out, iter_rows())
    return len(split), out


def export_qrels(dataset_name: str, outdir: Path, overwrite: bool):
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    counts = {}
    outputs = []

    for split_name in ds.keys():
        split = ds[split_name]
        out = outdir / f"qrels_{split_name}.jsonl"
        ensure_parent(out)
        fail_if_exists(out, overwrite)

        def iter_rows():
            for item in split:
                yield {
                    "query_id": item.get("query-id"),
                    "doc_id": item.get("corpus-id"),
                    "score": int(item.get("score", 0)),
                    "split": split_name,
                }

        write_jsonl(out, iter_rows())
        counts[split_name] = len(split)
        outputs.append(out)

    return counts, outputs


def export_from_hf(dataset: str, qrels_dataset: str, outdir: Path, overwrite: bool):
    corpus_n, corpus_out = export_corpus(dataset, outdir, overwrite)
    queries_n, queries_out = export_queries(dataset, outdir, overwrite)
    qrels_counts, _ = export_qrels(qrels_dataset, outdir, overwrite)
    return corpus_n, corpus_out, queries_n, queries_out, qrels_counts


def download_beir_zip(slug: str, tmp_dir: Path) -> Path:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{slug}.zip"
    zip_path = tmp_dir / f"{slug}.zip"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310
    return zip_path


def export_from_beir_zip(dataset: str, outdir: Path, overwrite: bool):
    slug = dataset_slug(dataset)
    tmp_root = outdir.parent / ".tmp_beir"
    zip_path = download_beir_zip(slug, tmp_root)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_root)

    root = tmp_root / slug
    src_corpus = root / "corpus.jsonl"
    src_queries = root / "queries.jsonl"
    src_qrels_dir = root / "qrels"

    corpus_out = outdir / "corpus.jsonl"
    queries_out = outdir / "queries.jsonl"
    fail_if_exists(corpus_out, overwrite)
    fail_if_exists(queries_out, overwrite)
    ensure_parent(corpus_out)
    ensure_parent(queries_out)
    shutil.copy2(src_corpus, corpus_out)
    shutil.copy2(src_queries, queries_out)

    corpus_n = sum(1 for _ in corpus_out.open("r", encoding="utf-8"))
    queries_n = sum(1 for _ in queries_out.open("r", encoding="utf-8"))

    qrels_counts = {}
    for tsv_path in sorted(src_qrels_dir.glob("*.tsv")):
        split_name = tsv_path.stem
        qrels_out = outdir / f"qrels_{split_name}.jsonl"
        fail_if_exists(qrels_out, overwrite)
        ensure_parent(qrels_out)

        with tsv_path.open("r", encoding="utf-8") as fin, qrels_out.open(
            "w", encoding="utf-8"
        ) as fout:
            reader = csv.DictReader(fin, delimiter="\t")
            n = 0
            for row in reader:
                item = {
                    "query_id": row.get("query-id"),
                    "doc_id": row.get("corpus-id"),
                    "score": int(row.get("score", 0)),
                    "split": split_name,
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        qrels_counts[split_name] = n

    return corpus_n, corpus_out, queries_n, queries_out, qrels_counts


def main():
    args = parse_args()
    slug = args.name or dataset_slug(args.dataset)
    outdir = Path(args.outdir) if args.outdir else Path("data") / slug / "raw"
    outdir.mkdir(parents=True, exist_ok=True)

    try_hf = args.source in {"auto", "hf"}
    use_zip = args.source == "beir-zip"

    if use_zip:
        corpus_n, corpus_out, queries_n, queries_out, qrels_counts = export_from_beir_zip(
            args.dataset, outdir, args.overwrite
        )
    else:
        try:
            corpus_n, corpus_out, queries_n, queries_out, qrels_counts = export_from_hf(
                args.dataset, args.qrels_dataset, outdir, args.overwrite
            )
        except RuntimeError as e:
            if try_hf and "Dataset scripts are no longer supported" in str(e):
                print("HF loader failed due dataset script deprecation, fallback to BEIR zip...")
                corpus_n, corpus_out, queries_n, queries_out, qrels_counts = export_from_beir_zip(
                    args.dataset, outdir, args.overwrite
                )
            else:
                raise

    print("Done.")
    print(f"corpus  : {corpus_n} -> {corpus_out}")
    print(f"queries : {queries_n} -> {queries_out}")
    for split_name, n in qrels_counts.items():
        print(f"qrels[{split_name}] : {n} -> {outdir / ('qrels_' + split_name + '.jsonl')}")

    print("\nNext step:")
    dataset_root = outdir.parent
    print(
        "python scripts/build_index.py "
        f"--dataset-dir {dataset_root} "
        f"--outdir {dataset_root / 'processed'}"
    )


if __name__ == "__main__":
    main()
