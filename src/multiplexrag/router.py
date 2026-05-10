from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from multiplexrag.eval_utils import mrr_at_k, score_metric


ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\d")
NON_ALNUM_RE = re.compile(r"[^\w\s]")
UPPER_RE = re.compile(r"[A-Z]")
WH_PREFIX_RE = re.compile(r"^\s*(what|which|who|when|where|why|how)\b", re.IGNORECASE)
COMPARATIVE_RE = re.compile(r"\b(vs|versus|compare|difference|better|best)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
RETRIEVAL_FEATURE_GROUPS = ("basic", "score_shape", "agreement_rich", "query_match")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "may",
    "might",
    "of",
    "on",
    "or",
    "should",
    "the",
    "through",
    "to",
    "under",
    "versus",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "would",
}


def query_features(text: str) -> dict[str, float]:
    tokens = text.split()
    lower_tokens = [token.lower() for token in tokens]
    unique_tokens = set(lower_tokens)
    punctuation = NON_ALNUM_RE.findall(text)
    digit_count = len(NUMBER_RE.findall(text))
    acronym_count = len(ACRONYM_RE.findall(text))
    upper_count = len(UPPER_RE.findall(text))
    stopword_count = sum(token in STOPWORDS for token in lower_tokens)
    long_token_count = sum(len(token) >= 8 for token in tokens)
    short_token_count = sum(len(token) <= 3 for token in tokens)
    alpha_count = sum(char.isalpha() for char in text)
    return {
        "char_len": float(len(text)),
        "token_len": float(len(tokens)),
        "avg_token_len": float(sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0,
        "unique_ratio": float(len(unique_tokens) / len(tokens)) if tokens else 0.0,
        "digit_count": float(digit_count),
        "digit_ratio": float(digit_count / max(len(text), 1)),
        "acronym_count": float(acronym_count),
        "acronym_ratio": float(acronym_count / max(len(tokens), 1)),
        "uppercase_ratio": float(upper_count / max(len(text), 1)),
        "has_question_mark": float("?" in text),
        "punctuation_count": float(len(punctuation)),
        "punctuation_ratio": float(len(punctuation) / max(len(text), 1)),
        "punctuation_variety": float(len(set(punctuation))),
        "stopword_ratio": float(stopword_count / max(len(tokens), 1)),
        "long_token_ratio": float(long_token_count / max(len(tokens), 1)),
        "short_token_ratio": float(short_token_count / max(len(tokens), 1)),
        "alpha_ratio": float(alpha_count / max(len(text), 1)),
        "contains_colon": float(":" in text),
        "contains_slash": float("/" in text),
        "contains_hyphen": float("-" in text),
        "contains_parentheses": float("(" in text or ")" in text),
        "starts_with_wh_word": float(bool(WH_PREFIX_RE.search(text))),
        "has_comparative_cue": float(bool(COMPARATIVE_RE.search(text))),
    }


def _score_series(payload: dict, n: int) -> list[float]:
    results = payload.get("results", []) if payload else []
    scores = [float(item.get("score", 0.0)) for item in results[:n]]
    if len(scores) < n:
        scores.extend([0.0] * (n - len(scores)))
    return scores


def _doc_id_series(payload: dict, n: int) -> list[str]:
    results = payload.get("results", []) if payload else []
    doc_ids = [str(item["doc_id"]) for item in results[:n]]
    if len(doc_ids) < n:
        doc_ids.extend([""] * (n - len(doc_ids)))
    return doc_ids


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-6 else 0.0


def _rank_of_doc(doc_ids: list[str], target_doc: str) -> float:
    if not target_doc:
        return 0.0
    try:
        return float(doc_ids.index(target_doc) + 1)
    except ValueError:
        return 0.0


def _token_set(text: str, max_tokens: int | None = None) -> set[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return set(tokens)


def _query_doc_match_feature_block(query_text: str, doc_text: str, prefix: str) -> dict[str, float]:
    query_tokens = _token_set(query_text)
    doc_tokens = _token_set(doc_text, max_tokens=64)
    overlap = query_tokens & doc_tokens
    long_query_tokens = {token for token in query_tokens if len(token) >= 5}
    long_overlap = long_query_tokens & doc_tokens
    return {
        f"{prefix}_overlap_count": float(len(overlap)),
        f"{prefix}_query_overlap_ratio": _safe_ratio(float(len(overlap)), float(len(query_tokens))),
        f"{prefix}_doc_overlap_ratio": _safe_ratio(float(len(overlap)), float(len(doc_tokens))),
        f"{prefix}_long_query_overlap_ratio": _safe_ratio(float(len(long_overlap)), float(len(long_query_tokens))),
    }


def retrieval_confidence_features(
    query_text: str,
    results_by_mode: dict[str, dict] | None,
    groups: tuple[str, ...] | None = None,
) -> dict[str, float]:
    modes = ("sparse", "dense", "hybrid")
    pairs = (("sparse", "dense"), ("sparse", "hybrid"), ("dense", "hybrid"))
    enabled = tuple(groups or RETRIEVAL_FEATURE_GROUPS)
    if not results_by_mode:
        empty = {}
        for mode in modes:
            if "basic" in enabled:
                empty[f"{mode}_top1_score"] = 0.0
                empty[f"{mode}_top1_gap"] = 0.0
                empty[f"{mode}_top1_gap_ratio"] = 0.0
                empty[f"{mode}_top3_mean_score"] = 0.0
                empty[f"{mode}_top3_std_score"] = 0.0
            if "score_shape" in enabled:
                empty[f"{mode}_top5_mean_score"] = 0.0
                empty[f"{mode}_top5_std_score"] = 0.0
                empty[f"{mode}_top10_mean_score"] = 0.0
                empty[f"{mode}_top10_std_score"] = 0.0
                empty[f"{mode}_top1_to_top3_mean_ratio"] = 0.0
                empty[f"{mode}_top1_to_top5_mean_ratio"] = 0.0
                empty[f"{mode}_top1_to_top10_mean_ratio"] = 0.0
            if "query_match" in enabled:
                empty[f"{mode}_top1_query_doc_overlap_count"] = 0.0
                empty[f"{mode}_top1_query_doc_query_overlap_ratio"] = 0.0
                empty[f"{mode}_top1_query_doc_doc_overlap_ratio"] = 0.0
                empty[f"{mode}_top1_query_doc_long_query_overlap_ratio"] = 0.0
        for left, right in pairs:
            if "basic" in enabled:
                empty[f"{left}_{right}_jaccard"] = 0.0
                empty[f"{left}_{right}_top1_same_doc"] = 0.0
            if "agreement_rich" in enabled:
                empty[f"{left}_{right}_overlap_count"] = 0.0
                empty[f"{left}_{right}_top3_overlap_count"] = 0.0
                empty[f"{left}_{right}_top5_overlap_count"] = 0.0
                empty[f"{left}_top1_rank_in_{right}"] = 0.0
                empty[f"{right}_top1_rank_in_{left}"] = 0.0
                empty[f"{left}_top1_recip_rank_in_{right}"] = 0.0
                empty[f"{right}_top1_recip_rank_in_{left}"] = 0.0
        return empty

    feats: dict[str, float] = {}
    doc_sets: dict[str, set[str]] = {}
    doc_lists: dict[str, list[str]] = {}
    top3_sets: dict[str, set[str]] = {}
    top5_sets: dict[str, set[str]] = {}
    top_docs: dict[str, str] = {}
    for mode in modes:
        payload = results_by_mode.get(mode, {})
        scores = _score_series(payload, 3)
        scores5 = _score_series(payload, 5)
        scores10 = _score_series(payload, 10)
        top1, top2 = scores[0], scores[1]
        if "basic" in enabled:
            feats[f"{mode}_top1_score"] = float(top1)
            feats[f"{mode}_top1_gap"] = float(top1 - top2)
            feats[f"{mode}_top1_gap_ratio"] = float((top1 - top2) / max(abs(top1), 1e-6))
            feats[f"{mode}_top3_mean_score"] = float(np.mean(scores))
            feats[f"{mode}_top3_std_score"] = float(np.std(scores))
        if "score_shape" in enabled:
            feats[f"{mode}_top5_mean_score"] = float(np.mean(scores5))
            feats[f"{mode}_top5_std_score"] = float(np.std(scores5))
            feats[f"{mode}_top10_mean_score"] = float(np.mean(scores10))
            feats[f"{mode}_top10_std_score"] = float(np.std(scores10))
            feats[f"{mode}_top1_to_top3_mean_ratio"] = _safe_ratio(top1, float(np.mean(scores)))
            feats[f"{mode}_top1_to_top5_mean_ratio"] = _safe_ratio(top1, float(np.mean(scores5)))
            feats[f"{mode}_top1_to_top10_mean_ratio"] = _safe_ratio(top1, float(np.mean(scores10)))
        if "query_match" in enabled:
            top_doc_text = ""
            top_result = payload.get("results", [])[:1]
            if top_result:
                top_doc_text = str(top_result[0].get("content", "") or "")
            feats.update(
                _query_doc_match_feature_block(
                    query_text,
                    top_doc_text,
                    f"{mode}_top1_query_doc",
                )
            )
        doc_ids = _doc_id_series(payload, 10)
        doc_sets[mode] = set(doc_id for doc_id in doc_ids if doc_id)
        doc_lists[mode] = doc_ids
        top3_sets[mode] = set(doc_id for doc_id in doc_ids[:3] if doc_id)
        top5_sets[mode] = set(doc_id for doc_id in doc_ids[:5] if doc_id)
        top_docs[mode] = doc_ids[0] if doc_ids else ""
    for left, right in pairs:
        union = doc_sets[left] | doc_sets[right]
        overlap = doc_sets[left] & doc_sets[right]
        if "basic" in enabled:
            feats[f"{left}_{right}_jaccard"] = float(len(overlap) / len(union)) if union else 0.0
            feats[f"{left}_{right}_top1_same_doc"] = float(bool(top_docs[left] and top_docs[left] == top_docs[right]))
        if "agreement_rich" in enabled:
            feats[f"{left}_{right}_overlap_count"] = float(len(overlap))
            feats[f"{left}_{right}_top3_overlap_count"] = float(len(top3_sets[left] & top3_sets[right]))
            feats[f"{left}_{right}_top5_overlap_count"] = float(len(top5_sets[left] & top5_sets[right]))
        left_rank_in_right = _rank_of_doc(doc_lists[right], top_docs[left])
        right_rank_in_left = _rank_of_doc(doc_lists[left], top_docs[right])
        if "agreement_rich" in enabled:
            feats[f"{left}_top1_rank_in_{right}"] = left_rank_in_right
            feats[f"{right}_top1_rank_in_{left}"] = right_rank_in_left
            feats[f"{left}_top1_recip_rank_in_{right}"] = _safe_ratio(1.0, left_rank_in_right)
            feats[f"{right}_top1_recip_rank_in_{left}"] = _safe_ratio(1.0, right_rank_in_left)
    return feats


def combined_features(
    text: str,
    results_by_mode: dict[str, dict] | None = None,
    *,
    include_retrieval_features: bool = False,
    retrieval_feature_groups: tuple[str, ...] | None = None,
) -> dict[str, float]:
    feats = query_features(text)
    if include_retrieval_features:
        feats.update(retrieval_confidence_features(text, results_by_mode, groups=retrieval_feature_groups))
    return feats


def features_matrix(
    queries: list[dict],
    query_results: dict[str, dict] | None = None,
    *,
    include_retrieval_features: bool = False,
    retrieval_feature_groups: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    feature_names = sorted(
        combined_features(
            "",
            include_retrieval_features=include_retrieval_features,
            retrieval_feature_groups=retrieval_feature_groups,
        ).keys()
    )
    rows = []
    for query in queries:
        feats = combined_features(
            query["text"],
            query_results.get(query["query_id"]) if query_results else None,
            include_retrieval_features=include_retrieval_features,
            retrieval_feature_groups=retrieval_feature_groups,
        )
        rows.append([feats[name] for name in feature_names])
    return np.asarray(rows, dtype=np.float32), feature_names


def mode_scores_for_query(
    results_by_mode: dict[str, list[str]],
    relevant: set[str],
    metrics: list[str] | None = None,
    weights: list[float] | None = None,
    *,
    k: int = 10,
) -> dict[str, dict[str, float]]:
    metric_names = metrics or [f"mrr@{k}"]
    metric_weights = weights or [1.0] * len(metric_names)
    if len(metric_names) != len(metric_weights):
        raise ValueError("metrics and weights must have the same length")
    per_mode: dict[str, dict[str, float]] = {}
    for mode, doc_ids in results_by_mode.items():
        metric_values = {
            metric: score_metric(metric, doc_ids, relevant)
            for metric in metric_names
        }
        combined = sum(
            metric_values[metric] * weight
            for metric, weight in zip(metric_names, metric_weights)
        )
        per_mode[mode] = {
            **metric_values,
            "combined": float(combined),
        }
    return per_mode


def best_mode_for_query(
    results_by_mode: dict[str, list[str]],
    relevant: set[str],
    k: int = 10,
    metrics: list[str] | None = None,
    weights: list[float] | None = None,
    tie_preference: str = "hybrid",
    near_tie_mode: str | None = None,
    near_tie_margin: float = 0.0,
) -> str:
    scored = mode_scores_for_query(
        results_by_mode,
        relevant,
        metrics=metrics,
        weights=weights,
        k=k,
    )
    best_mode, best_payload = max(
        scored.items(),
        key=lambda item: (item[1]["combined"], item[0] == tie_preference),
    )
    if near_tie_mode and near_tie_mode in scored:
        preferred_score = scored[near_tie_mode]["combined"]
        if (best_payload["combined"] - preferred_score) <= float(near_tie_margin):
            return near_tie_mode
    return best_mode


class QueryRouter:
    def __init__(
        self,
        *,
        min_confidence: float = 0.45,
        fallback_mode: str = "dense",
        word_features: int = 2**12,
        char_features: int = 2**12,
        use_retrieval_features: bool = False,
        retrieval_feature_groups: tuple[str, ...] | None = None,
        random_state: int = 42,
    ) -> None:
        self.feature_names: list[str] = []
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.min_confidence = float(min_confidence)
        self.fallback_mode = fallback_mode
        self.word_features = int(word_features)
        self.char_features = int(char_features)
        self.use_retrieval_features = bool(use_retrieval_features)
        self.retrieval_feature_groups = tuple(retrieval_feature_groups or RETRIEVAL_FEATURE_GROUPS)
        self.random_state = int(random_state)
        self.word_vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=self.word_features,
            alternate_sign=False,
            lowercase=True,
            norm="l2",
        )
        self.char_vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=self.char_features,
            alternate_sign=False,
            lowercase=True,
            norm="l2",
        )
        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            random_state=self.random_state,
        )
        self.labels_: list[str] = []

    def _fit_numeric_stats(self, x_numeric: np.ndarray) -> np.ndarray:
        self.numeric_mean = x_numeric.mean(axis=0)
        self.numeric_std = x_numeric.std(axis=0)
        self.numeric_std[self.numeric_std == 0.0] = 1.0
        return (x_numeric - self.numeric_mean) / self.numeric_std

    def _transform_numeric(self, x_numeric: np.ndarray) -> np.ndarray:
        if self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("QueryRouter numeric statistics are not fitted.")
        return (x_numeric - self.numeric_mean) / self.numeric_std

    def _vectorize_queries(
        self,
        queries: list[dict],
        query_results: dict[str, dict] | None = None,
        *,
        fit: bool,
    ) -> sparse.csr_matrix:
        x_numeric, feature_names = features_matrix(
            queries,
            query_results,
            include_retrieval_features=self.use_retrieval_features,
            retrieval_feature_groups=self.retrieval_feature_groups,
        )
        if fit:
            x_numeric = self._fit_numeric_stats(x_numeric)
            self.feature_names = feature_names
        else:
            x_numeric = self._transform_numeric(x_numeric)
        texts = [query["text"] for query in queries]
        numeric_block = sparse.csr_matrix(x_numeric)
        word_block = self.word_vectorizer.transform(texts)
        char_block = self.char_vectorizer.transform(texts)
        return sparse.hstack([numeric_block, word_block, char_block], format="csr")

    def fit(
        self,
        queries: list[dict],
        labels: list[str],
        query_results: dict[str, dict] | None = None,
    ) -> "QueryRouter":
        x = self._vectorize_queries(queries, query_results, fit=True)
        self.classifier.fit(x, labels)
        self.labels_ = list(self.classifier.classes_)
        return self

    def predict(
        self,
        queries: list[dict],
        query_results: dict[str, dict] | None = None,
    ) -> list[str]:
        if not queries:
            return []
        x = self._vectorize_queries(queries, query_results, fit=False)
        labels = self.classifier.predict(x)
        probs = self.classifier.predict_proba(x)
        confidences = probs.max(axis=1)
        preds = []
        for label, confidence in zip(labels, confidences):
            if confidence < self.min_confidence and self.fallback_mode in self.labels_:
                preds.append(self.fallback_mode)
                continue
            preds.append(str(label))
        return preds

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(
                {
                    "feature_names": self.feature_names,
                    "numeric_mean": self.numeric_mean,
                    "numeric_std": self.numeric_std,
                    "classifier": self.classifier,
                    "min_confidence": self.min_confidence,
                    "fallback_mode": self.fallback_mode,
                    "word_features": self.word_features,
                    "char_features": self.char_features,
                    "use_retrieval_features": self.use_retrieval_features,
                    "retrieval_feature_groups": self.retrieval_feature_groups,
                    "random_state": self.random_state,
                    "labels_": self.labels_,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "QueryRouter":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        obj = cls(
            min_confidence=payload.get("min_confidence", 0.45),
            fallback_mode=payload.get("fallback_mode", "dense"),
            word_features=payload.get("word_features", 2**12),
            char_features=payload.get("char_features", 2**12),
            use_retrieval_features=payload.get("use_retrieval_features", False),
            retrieval_feature_groups=tuple(payload.get("retrieval_feature_groups", RETRIEVAL_FEATURE_GROUPS)),
            random_state=payload.get("random_state", 42),
        )
        obj.feature_names = payload["feature_names"]
        obj.numeric_mean = payload.get("numeric_mean")
        obj.numeric_std = payload.get("numeric_std")
        obj.classifier = payload["classifier"]
        obj.labels_ = payload.get("labels_", list(obj.classifier.classes_))
        return obj
