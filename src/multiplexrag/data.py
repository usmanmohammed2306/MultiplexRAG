from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_corpus_row(row: dict) -> dict:
    doc_id = str(row.get("doc_id", row.get("_id", "")))
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "content": f"{title}. {text}".strip(". ").strip(),
        "metadata": row.get("metadata", {}),
    }


def normalize_query_row(row: dict) -> dict:
    query_id = str(row.get("query_id", row.get("_id", "")))
    return {
        "query_id": query_id,
        "text": row.get("text", "") or "",
        "metadata": row.get("metadata", {}),
    }


def load_corpus(path: str | Path) -> list[dict]:
    return [normalize_corpus_row(row) for row in load_jsonl(path)]


def load_queries(path: str | Path) -> list[dict]:
    return [normalize_query_row(row) for row in load_jsonl(path)]


def load_qrels(path: str | Path, min_score: int = 1) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in load_jsonl(path):
        score = int(row.get("score", 0))
        if score < min_score:
            continue
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def approximate_token_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len(stripped.split())


def corpus_token_counts(corpus: list[dict]) -> dict[str, int]:
    return {
        str(doc["doc_id"]): approximate_token_count(doc.get("content", ""))
        for doc in corpus
    }
