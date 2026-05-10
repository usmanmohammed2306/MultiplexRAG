#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Analyze router errors against dense baseline.")
    p.add_argument("--dataset", default="scidocs", help="Dataset name under data/ and phase1/.")
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    from multiplexrag.data import load_qrels, load_queries
    from multiplexrag.eval_utils import mrr_at_k
    from multiplexrag.router import query_features

    args = parse_args()
    dataset = args.dataset

    queries = {row["query_id"]: row["text"] for row in load_queries(ROOT / f"data/{dataset}/raw/queries.jsonl")}
    if dataset == "scidocs":
        test_qrels_path = ROOT / "data/scidocs/raw/qrels_router_test.jsonl"
    else:
        test_qrels_path = ROOT / f"data/{dataset}/raw/qrels_test.jsonl"
    test_qrels = load_qrels(test_qrels_path)

    router_predictions = {
        row["query_id"]: row
        for row in load_jsonl(ROOT / f"phase1/{dataset}/router_predictions.jsonl")
    }
    by_mode = {}
    for mode in ("sparse", "dense", "hybrid"):
        by_mode[mode] = {
            row["query_id"]: row
            for row in load_jsonl(ROOT / f"phase1/{dataset}/{mode}.jsonl")
        }

    rows = []
    for query_id, relevant in test_qrels.items():
        query = queries[query_id]
        router_row = router_predictions[query_id]
        dense_docs = [item["doc_id"] for item in by_mode["dense"][query_id]["results"]]
        router_docs = [item["doc_id"] for item in router_row["results"]]
        scores = {
            mode: mrr_at_k([item["doc_id"] for item in by_mode[mode][query_id]["results"]], relevant, args.topk)
            for mode in ("sparse", "dense", "hybrid")
        }
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "predicted_mode": router_row["predicted_mode"],
                "gold_mode": router_row["gold_mode"],
                "dense_mrr": mrr_at_k(dense_docs, relevant, args.topk),
                "router_mrr": mrr_at_k(router_docs, relevant, args.topk),
                "delta_vs_dense": mrr_at_k(router_docs, relevant, args.topk) - mrr_at_k(dense_docs, relevant, args.topk),
                "scores": scores,
                **query_features(query),
            }
        )

    print(f"Queries analyzed: {len(rows)}")
    print(f"Dense avg MRR@{args.topk}: {sum(row['dense_mrr'] for row in rows) / len(rows):.6f}")
    print(f"Router avg MRR@{args.topk}: {sum(row['router_mrr'] for row in rows) / len(rows):.6f}")
    print(f"Average delta (router - dense): {sum(row['delta_vs_dense'] for row in rows) / len(rows):.6f}")
    print()

    print("Prediction counts:", dict(Counter(row["predicted_mode"] for row in rows)))
    print("Gold counts:", dict(Counter(row["gold_mode"] for row in rows)))
    print()

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["gold_mode"], row["predicted_mode"])].append(row["delta_vs_dense"])
    print("Mean delta vs dense by (gold, predicted):")
    for key, values in sorted(grouped.items()):
        print(f"  {key}: n={len(values)}, mean_delta={sum(values) / len(values):.6f}")
    print()

    ties = 0
    zero_ties = 0
    for row in rows:
        scores = row["scores"]
        if scores["dense"] == scores["hybrid"] and scores["dense"] >= scores["sparse"]:
            ties += 1
        if scores["dense"] == scores["hybrid"] == scores["sparse"] == 0.0:
            zero_ties += 1
    print(f"Dense/hybrid top ties: {ties}")
    print(f"All-zero ties: {zero_ties}")
    print()

    worst = sorted(rows, key=lambda row: row["delta_vs_dense"])[:10]
    print("Worst losses vs dense:")
    for row in worst:
        print(f"  {row['delta_vs_dense']:.6f} | gold={row['gold_mode']} pred={row['predicted_mode']} | {row['query']}")


if __name__ == "__main__":
    main()
