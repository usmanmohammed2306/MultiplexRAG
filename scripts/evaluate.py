#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval/router outputs.")
    p.add_argument("--dataset-dir", default="", help="Dataset root containing raw/ folder")
    p.add_argument("--pred", default="results/retrieval.jsonl")
    p.add_argument("--qrels", default="data/raw/qrels_test.jsonl")
    p.add_argument("--metrics", nargs="+", default=["recall@10", "mrr@10", "ndcg@10"])
    p.add_argument("--out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    from multiplexrag.data import load_jsonl, load_qrels
    from multiplexrag.eval_utils import score_metric

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    qrels_path = dataset_dir / "raw" / "qrels_test.jsonl" if dataset_dir else Path(args.qrels)
    predictions = load_jsonl(args.pred)
    qrels = load_qrels(qrels_path)

    totals = {metric: 0.0 for metric in args.metrics}
    counted = 0
    latencies = []
    estimated_tokens = []
    estimated_costs = []
    for row in predictions:
        query_id = str(row["query_id"])
        if query_id not in qrels:
            continue
        retrieved = [item["doc_id"] for item in row.get("results", [])]
        relevant = qrels[query_id]
        for metric in args.metrics:
            totals[metric] += score_metric(metric, retrieved, relevant)
        latencies.append(float(row.get("latency_ms", 0.0)))
        estimated_tokens.append(float(row.get("estimated_total_tokens", 0.0)))
        estimated_costs.append(float(row.get("estimated_cost_usd", 0.0)))
        counted += 1

    summary = {metric: (totals[metric] / counted if counted else 0.0) for metric in args.metrics}
    summary["queries_evaluated"] = counted
    summary["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
    summary["avg_estimated_total_tokens"] = (
        sum(estimated_tokens) / len(estimated_tokens) if estimated_tokens else 0.0
    )
    summary["avg_estimated_cost_usd"] = (
        sum(estimated_costs) / len(estimated_costs) if estimated_costs else 0.0
    )
    summary["pred_path"] = args.pred
    summary["qrels_path"] = str(qrels_path)

    print(json.dumps(summary, indent=2))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary -> {target}")


if __name__ == "__main__":
    main()
