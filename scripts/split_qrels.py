#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a query-level train/test split from an existing qrels file."
    )
    p.add_argument("--qrels", required=True, help="Input qrels jsonl file.")
    p.add_argument("--train-out", required=True, help="Output train qrels jsonl file.")
    p.add_argument("--test-out", required=True, help="Output test qrels jsonl file.")
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of unique query ids assigned to train.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for query-level split.")
    p.add_argument(
        "--meta-out",
        default="",
        help="Optional JSON metadata file describing the split.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    qrels_path = Path(args.qrels)
    rows = load_jsonl(qrels_path)
    query_ids = sorted({str(row["query_id"]) for row in rows})

    rng = random.Random(args.seed)
    shuffled = query_ids[:]
    rng.shuffle(shuffled)

    split_idx = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * args.train_ratio))))
    train_queries = set(shuffled[:split_idx])
    test_queries = set(shuffled[split_idx:])

    train_rows = [row for row in rows if str(row["query_id"]) in train_queries]
    test_rows = [row for row in rows if str(row["query_id"]) in test_queries]

    train_out = Path(args.train_out)
    test_out = Path(args.test_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(test_out, test_rows)

    meta = {
        "source_qrels": str(qrels_path),
        "split_unit": "query_id",
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total_qrels_rows": len(rows),
        "train_qrels_rows": len(train_rows),
        "test_qrels_rows": len(test_rows),
        "total_unique_queries": len(query_ids),
        "train_unique_queries": len(train_queries),
        "test_unique_queries": len(test_queries),
        "train_out": str(train_out),
        "test_out": str(test_out),
    }

    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Input qrels         : {qrels_path}")
    print(f"Train qrels output  : {train_out}")
    print(f"Test qrels output   : {test_out}")
    print(f"Unique queries      : {len(query_ids)}")
    print(f"Train queries       : {len(train_queries)}")
    print(f"Test queries        : {len(test_queries)}")
    print(f"Train qrels rows    : {len(train_rows)}")
    print(f"Test qrels rows     : {len(test_rows)}")
    if args.meta_out:
        print(f"Metadata            : {args.meta_out}")


if __name__ == "__main__":
    main()
