#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Train query router for MultiplexRAG.")
    p.add_argument("--dataset-dir", default="", help="Dataset root containing raw/ and processed/ folders")
    p.add_argument("--queries", default="data/raw/queries.jsonl")
    p.add_argument("--corpus", default="data/raw/corpus.jsonl")
    p.add_argument("--train-qrels", default="data/raw/qrels_train.jsonl")
    p.add_argument("--test-qrels", default="data/raw/qrels_test.jsonl")
    p.add_argument("--model-out", default="results/router.pkl")
    p.add_argument("--report-out", default="results/router_report.json")
    p.add_argument("--pred-out", default="results/router_predictions.jsonl")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--policy", choices=["classifier"], default="classifier")
    p.add_argument(
        "--label-metrics",
        nargs="+",
        default=[],
        help="Metrics used to create weak router labels, for example: mrr@10 ndcg@10 recall@10",
    )
    p.add_argument(
        "--label-weights",
        nargs="+",
        type=float,
        default=[],
        help="Weights aligned with --label-metrics. Defaults to equal weights.",
    )
    p.add_argument(
        "--label-tie-preference",
        choices=["sparse", "dense", "hybrid"],
        default="hybrid",
        help="Which mode wins when the label metrics tie.",
    )
    p.add_argument(
        "--label-near-tie-mode",
        choices=["sparse", "dense", "hybrid"],
        default="",
        help="Optional preferred mode when its weak-label score is within the configured margin of the best mode.",
    )
    p.add_argument(
        "--label-near-tie-margin",
        type=float,
        default=0.0,
        help="If best_score - preferred_mode_score <= margin, assign the preferred mode label instead.",
    )
    p.add_argument(
        "--use-retrieval-features",
        action="store_true",
        help="Augment query-only router features with retrieval-confidence signals from sparse/dense/hybrid results.",
    )
    p.add_argument(
        "--retrieval-feature-groups",
        nargs="+",
        choices=["basic", "score_shape", "agreement_rich", "query_match"],
        default=[],
        help="Optional retrieval-feature groups to enable. Defaults to all groups when --use-retrieval-features is set.",
    )
    p.add_argument(
        "--router-confidence-threshold",
        type=float,
        default=0.45,
        help="Fallback to the configured router mode when predicted confidence is below this threshold.",
    )
    p.add_argument(
        "--router-fallback-mode",
        choices=["sparse", "dense", "hybrid"],
        default="dense",
        help="Fallback strategy used when the router is uncertain.",
    )
    p.add_argument(
        "--usd-per-1k-tokens",
        type=float,
        default=0.002,
        help="Estimated downstream input price in USD per 1K tokens for query + retrieved context.",
    )
    return p.parse_args()


def collect_query_results(
    queries: list[dict],
    retrievers: dict,
    corpus: list[dict],
    topk: int,
    usd_per_1k_tokens: float,
) -> dict[str, dict]:
    from multiplexrag.data import approximate_token_count

    collected: dict[str, dict] = {}
    content_by_doc_id = {
        str(doc["doc_id"]): str(doc.get("content", "") or "")
        for doc in corpus
    }
    for query in queries:
        by_mode = {}
        query_tokens = approximate_token_count(query["text"])
        for mode, retriever in retrievers.items():
            results, latency_ms = retriever.search(query["text"], topk=topk)
            enriched_results = []
            for item in results:
                row = dict(item)
                row["content"] = content_by_doc_id.get(str(item["doc_id"]), "")
                enriched_results.append(row)
            context_tokens = sum(int(item.get("token_count", 0)) for item in results)
            total_tokens = query_tokens + context_tokens
            by_mode[mode] = {
                "results": enriched_results,
                "doc_ids": [item["doc_id"] for item in enriched_results],
                "latency_ms": float(latency_ms),
                "estimated_total_tokens": float(total_tokens),
                "estimated_cost_usd": float(total_tokens / 1000.0 * usd_per_1k_tokens),
            }
        collected[query["query_id"]] = by_mode
    return collected


def label_queries(
    queries: list[dict],
    qrels: dict[str, set[str]],
    query_results: dict[str, dict],
    topk: int,
    label_metrics: list[str] | None = None,
    label_weights: list[float] | None = None,
    tie_preference: str = "hybrid",
    near_tie_mode: str = "",
    near_tie_margin: float = 0.0,
) -> tuple[list[dict], list[str]]:
    from multiplexrag.router import best_mode_for_query

    kept_queries = []
    labels = []
    for query in queries:
        relevant = qrels.get(query["query_id"])
        if not relevant:
            continue
        results_by_mode = {
            mode: payload["doc_ids"]
            for mode, payload in query_results[query["query_id"]].items()
        }
        kept_queries.append(query)
        labels.append(
            best_mode_for_query(
                results_by_mode,
                relevant,
                k=topk,
                metrics=label_metrics,
                weights=label_weights,
                tie_preference=tie_preference,
                near_tie_mode=near_tie_mode or None,
                near_tie_margin=near_tie_margin,
            )
        )
    return kept_queries, labels


def summarize_metrics(
    metrics: list[str],
    chosen_doc_ids: dict[str, list[str]],
    chosen_latencies: dict[str, float],
    chosen_tokens: dict[str, float],
    chosen_costs: dict[str, float],
    qrels: dict[str, set[str]],
) -> dict[str, float]:
    from multiplexrag.eval_utils import score_metric

    summary = {metric: 0.0 for metric in metrics}
    counted = 0
    latency_total = 0.0
    token_total = 0.0
    cost_total = 0.0
    for query_id, relevant in qrels.items():
        doc_ids = chosen_doc_ids.get(query_id, [])
        for metric in metrics:
            summary[metric] += score_metric(metric, doc_ids, relevant)
        latency_total += chosen_latencies.get(query_id, 0.0)
        token_total += chosen_tokens.get(query_id, 0.0)
        cost_total += chosen_costs.get(query_id, 0.0)
        counted += 1
    if counted == 0:
        return {
            **summary,
            "queries_evaluated": 0,
            "avg_latency_ms": 0.0,
            "avg_estimated_total_tokens": 0.0,
            "avg_estimated_cost_usd": 0.0,
        }
    for metric in metrics:
        summary[metric] /= counted
    summary["queries_evaluated"] = counted
    summary["avg_latency_ms"] = latency_total / counted
    summary["avg_estimated_total_tokens"] = token_total / counted
    summary["avg_estimated_cost_usd"] = cost_total / counted
    return summary


def choose_mode_outputs(
    query_results: dict[str, dict],
    qrels: dict[str, set[str]],
    selection: dict[str, str],
    metrics: list[str],
) -> dict[str, float]:
    chosen_doc_ids = {}
    chosen_latencies = {}
    chosen_tokens = {}
    chosen_costs = {}
    for query_id in qrels:
        mode = selection[query_id]
        chosen_doc_ids[query_id] = query_results[query_id][mode]["doc_ids"]
        chosen_latencies[query_id] = query_results[query_id][mode]["latency_ms"]
        chosen_tokens[query_id] = query_results[query_id][mode]["estimated_total_tokens"]
        chosen_costs[query_id] = query_results[query_id][mode]["estimated_cost_usd"]
    return summarize_metrics(
        metrics,
        chosen_doc_ids,
        chosen_latencies,
        chosen_tokens,
        chosen_costs,
        qrels,
    )


def main():
    args = parse_args()
    from multiplexrag.data import load_corpus, load_queries, load_qrels
    from multiplexrag.retrieval import DenseRetriever, HybridRetriever, SparseRetriever, load_pickle
    from multiplexrag.router import QueryRouter, best_mode_for_query

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    corpus_path = (
        dataset_dir / "raw" / "corpus.jsonl"
        if dataset_dir and args.corpus == "data/raw/corpus.jsonl"
        else Path(args.corpus)
    )
    queries_path = (
        dataset_dir / "raw" / "queries.jsonl"
        if dataset_dir and args.queries == "data/raw/queries.jsonl"
        else Path(args.queries)
    )
    train_qrels_path = (
        dataset_dir / "raw" / "qrels_train.jsonl"
        if dataset_dir and args.train_qrels == "data/raw/qrels_train.jsonl"
        else Path(args.train_qrels)
    )
    test_qrels_path = (
        dataset_dir / "raw" / "qrels_test.jsonl"
        if dataset_dir and args.test_qrels == "data/raw/qrels_test.jsonl"
        else Path(args.test_qrels)
    )

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    train_qrels = load_qrels(train_qrels_path)
    test_qrels = load_qrels(test_qrels_path)
    metrics = [f"recall@{args.topk}", f"mrr@{args.topk}", f"ndcg@{args.topk}"]
    label_metrics = args.label_metrics or [f"mrr@{args.topk}"]
    label_weights = args.label_weights or [1.0] * len(label_metrics)
    if len(label_metrics) != len(label_weights):
        raise ValueError("--label-metrics and --label-weights must have the same length")

    processed_dir = dataset_dir / "processed" if dataset_dir else None
    sparse_index_path = processed_dir / "sparse.pkl" if processed_dir else None
    dense_index_path = processed_dir / "dense.pkl" if processed_dir else None

    if sparse_index_path and sparse_index_path.exists():
        sparse = load_pickle(sparse_index_path)
    else:
        sparse = SparseRetriever().fit(corpus)

    if dense_index_path and dense_index_path.exists():
        dense = load_pickle(dense_index_path)
    else:
        dense = DenseRetriever().fit(corpus)

    hybrid = HybridRetriever(sparse, dense)
    retrievers = {"sparse": sparse, "dense": dense, "hybrid": hybrid}

    query_results = collect_query_results(queries, retrievers, corpus, args.topk, args.usd_per_1k_tokens)
    train_queries, train_labels = label_queries(
        queries,
        train_qrels,
        query_results,
        args.topk,
        label_metrics=label_metrics,
        label_weights=label_weights,
        tie_preference=args.label_tie_preference,
        near_tie_mode=args.label_near_tie_mode,
        near_tie_margin=args.label_near_tie_margin,
    )
    test_queries, test_labels = label_queries(
        queries,
        test_qrels,
        query_results,
        args.topk,
        label_metrics=label_metrics,
        label_weights=label_weights,
        tie_preference=args.label_tie_preference,
        near_tie_mode=args.label_near_tie_mode,
        near_tie_margin=args.label_near_tie_margin,
    )
    train_query_results = {query["query_id"]: query_results[query["query_id"]] for query in train_queries}
    test_query_results = {query["query_id"]: query_results[query["query_id"]] for query in test_queries}

    router = QueryRouter(
        min_confidence=args.router_confidence_threshold,
        fallback_mode=args.router_fallback_mode,
        use_retrieval_features=args.use_retrieval_features,
        retrieval_feature_groups=tuple(args.retrieval_feature_groups),
    ).fit(train_queries, train_labels, train_query_results)
    preds = router.predict(test_queries, test_query_results)
    labels = sorted(set(train_labels) | set(test_labels))
    test_query_ids = [query["query_id"] for query in test_queries]
    gold_selection = {
        query_id: gold
        for query_id, gold in zip(test_query_ids, test_labels)
    }
    router_selection = {
        query_id: pred
        for query_id, pred in zip(test_query_ids, preds)
    }
    oracle_selection = {}
    for query in test_queries:
        query_id = query["query_id"]
        relevant = test_qrels[query_id]
        results_by_mode = {
            mode: payload["doc_ids"]
            for mode, payload in query_results[query_id].items()
        }
        oracle_selection[query_id] = best_mode_for_query(results_by_mode, relevant, k=args.topk)

    baseline_metrics = {}
    for mode in ("sparse", "dense", "hybrid"):
        selection = {query_id: mode for query_id in test_qrels}
        baseline_metrics[mode] = choose_mode_outputs(query_results, test_qrels, selection, metrics)
    router_metrics = choose_mode_outputs(query_results, test_qrels, router_selection, metrics)
    oracle_metrics = choose_mode_outputs(query_results, test_qrels, oracle_selection, metrics)

    prediction_rows = []
    for query in test_queries:
        query_id = query["query_id"]
        chosen_mode = router_selection[query_id]
        chosen = query_results[query_id][chosen_mode]
        prediction_rows.append(
            {
                "query_id": query_id,
                "query": query["text"],
                "predicted_mode": chosen_mode,
                "oracle_mode": oracle_selection[query_id],
                "gold_mode": gold_selection[query_id],
                "latency_ms": chosen["latency_ms"],
                "estimated_total_tokens": chosen["estimated_total_tokens"],
                "estimated_cost_usd": chosen["estimated_cost_usd"],
                "results": chosen["results"],
            }
        )

    report = {
        "accuracy": (
            sum(int(gold == pred) for gold, pred in zip(test_labels, preds)) / len(test_labels)
            if test_labels
            else 0.0
        ),
        "labels": labels,
        "router_settings": {
            "policy": args.policy,
            "confidence_threshold": args.router_confidence_threshold,
            "fallback_mode": args.router_fallback_mode,
            "use_retrieval_features": args.use_retrieval_features,
            "retrieval_feature_groups": args.retrieval_feature_groups,
            "label_metrics": label_metrics,
            "label_weights": label_weights,
            "label_tie_preference": args.label_tie_preference,
            "label_near_tie_mode": args.label_near_tie_mode or None,
            "label_near_tie_margin": args.label_near_tie_margin,
        },
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "train_label_distribution": dict(sorted(Counter(train_labels).items())),
        "test_label_distribution": dict(sorted(Counter(test_labels).items())),
        "predicted_label_distribution": dict(sorted(Counter(preds).items())),
        "metrics": metrics,
        "baselines": baseline_metrics,
        "router": router_metrics,
        "oracle": oracle_metrics,
    }

    router.save(args.model_out)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    from multiplexrag.data import write_jsonl

    write_jsonl(args.pred_out, prediction_rows)

    label_counts = {label: train_labels.count(label) for label in sorted(set(train_labels))}
    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries : {len(test_queries)}")
    print(f"Train label distribution: {label_counts}")
    print("\nComparison:")
    for name, payload in (
        ("sparse", baseline_metrics["sparse"]),
        ("dense", baseline_metrics["dense"]),
        ("hybrid", baseline_metrics["hybrid"]),
        ("router", router_metrics),
        ("oracle", oracle_metrics),
    ):
        metric_str = ", ".join(
            f"{metric}={payload[metric]:.4f}" for metric in metrics
        )
        print(
            f"  {name:>6}: {metric_str}, latency={payload['avg_latency_ms']:.2f} ms, "
            f"tokens={payload['avg_estimated_total_tokens']:.2f}, "
            f"cost=${payload['avg_estimated_cost_usd']:.6f}"
        )
    print(json.dumps(report, indent=2))
    print(f"Saved model  -> {args.model_out}")
    print(f"Saved report -> {args.report_out}")
    print(f"Saved preds  -> {args.pred_out}")


if __name__ == "__main__":
    main()
