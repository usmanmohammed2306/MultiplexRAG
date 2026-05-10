#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize corpus, query, and qrels statistics for one or more datasets."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "scidocs", "nfcorpus", "fiqa"],
        help="Datasets under data/ to summarize.",
    )
    p.add_argument(
        "--json-out",
        default=str(ROOT / "results" / "dataset_stats.json"),
        help="Where to write the machine-readable summary.",
    )
    p.add_argument(
        "--md-out",
        default=str(ROOT / "results" / "dataset_stats.md"),
        help="Where to write the markdown table.",
    )
    return p.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def corpus_stats(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    sizes = []
    for row in iter_jsonl(path):
        text = row.get("text") or row.get("content") or ""
        title = row.get("title") or ""
        sizes.append(len(TOKEN_RE.findall(f"{title} {text}")))
    if not sizes:
        return {"docs": 0, "avg_tokens": 0.0, "median_tokens": 0.0}
    return {
        "docs": len(sizes),
        "avg_tokens": round(sum(sizes) / len(sizes), 2),
        "median_tokens": float(statistics.median(sizes)),
    }


def query_stats(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    char_lens = []
    token_lens = []
    for row in iter_jsonl(path):
        text = row.get("text") or row.get("content") or ""
        char_lens.append(len(text))
        token_lens.append(len(TOKEN_RE.findall(text)))
    if not token_lens:
        return {"queries": 0}
    return {
        "queries": len(token_lens),
        "avg_chars": round(sum(char_lens) / len(char_lens), 2),
        "avg_tokens": round(sum(token_lens) / len(token_lens), 2),
        "median_tokens": float(statistics.median(token_lens)),
    }


def qrels_stats(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    per_query: dict[str, int] = {}
    rows = 0
    for row in iter_jsonl(path):
        qid = str(row.get("query_id"))
        per_query[qid] = per_query.get(qid, 0) + 1
        rows += 1
    if not per_query:
        return {"rows": 0, "unique_queries": 0}
    counts = list(per_query.values())
    return {
        "rows": rows,
        "unique_queries": len(per_query),
        "avg_rels_per_query": round(sum(counts) / len(counts), 2),
        "median_rels_per_query": float(statistics.median(counts)),
    }


def collect(dataset: str) -> dict:
    raw = DATA_DIR / dataset / "raw"
    return {
        "dataset": dataset,
        "raw_dir": str(raw.relative_to(ROOT)) if raw.exists() else None,
        "corpus": corpus_stats(raw / "corpus.jsonl"),
        "queries": query_stats(raw / "queries.jsonl"),
        "qrels_train": qrels_stats(raw / "qrels_train.jsonl"),
        "qrels_dev": qrels_stats(raw / "qrels_dev.jsonl"),
        "qrels_test": qrels_stats(raw / "qrels_test.jsonl"),
        "qrels_router_train": qrels_stats(raw / "qrels_router_train.jsonl"),
        "qrels_router_test": qrels_stats(raw / "qrels_router_test.jsonl"),
    }


def fmt(value):
    if value is None:
        return "-"
    return str(value)


def render_markdown(rows: list[dict]) -> str:
    header = (
        "| Dataset | Corpus docs | Avg doc tokens | Queries | Avg query tokens | "
        "Train qrels | Dev qrels | Test qrels | Avg rels/query (test) |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for row in rows:
        corpus = row.get("corpus") or {}
        queries = row.get("queries") or {}
        train = row.get("qrels_train") or {}
        dev = row.get("qrels_dev") or {}
        test = row.get("qrels_test") or {}
        lines.append(
            "| {ds} | {docs} | {dtok} | {qs} | {qtok} | {trn} | {dv} | {tst} | {avgr} |".format(
                ds=row["dataset"],
                docs=fmt(corpus.get("docs")),
                dtok=fmt(corpus.get("avg_tokens")),
                qs=fmt(queries.get("queries")),
                qtok=fmt(queries.get("avg_tokens")),
                trn=fmt(train.get("rows")),
                dv=fmt(dev.get("rows")),
                tst=fmt(test.get("rows")),
                avgr=fmt(test.get("avg_rels_per_query")),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    rows = [collect(name) for name in args.datasets]

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {json_out.relative_to(ROOT)}")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(rows), encoding="utf-8")
    print(f"saved {md_out.relative_to(ROOT)}")

    print()
    print(render_markdown(rows))


if __name__ == "__main__":
    main()
