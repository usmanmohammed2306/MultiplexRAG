from __future__ import annotations

import pickle
import re
import time
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from multiplexrag.data import approximate_token_count


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def stable_hash(text: str, dim: int) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) % (2**32)
    return value % dim


def char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    compact = SPACE_RE.sub(" ", text.lower().strip())
    grams = []
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            grams.append(compact[i : i + n])
    return grams


def _topk_from_scores(
    doc_ids: list[str],
    scores: np.ndarray,
    topk: int,
    doc_token_counts: dict[str, int] | None = None,
) -> list[dict]:
    if len(doc_ids) == 0:
        return []
    topk = min(topk, len(doc_ids))
    order = np.argpartition(-scores, topk - 1)[:topk]
    order = order[np.argsort(-scores[order])]
    results = []
    for rank, idx in enumerate(order, start=1):
        doc_id = doc_ids[idx]
        item = {"doc_id": doc_id, "score": float(scores[idx]), "rank": rank}
        if doc_token_counts is not None:
            item["token_count"] = int(doc_token_counts.get(str(doc_id), 0))
        results.append(item)
    return results


class SparseRetriever:
    kind = "sparse"

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self.doc_token_counts: dict[str, int] = {}
        self.bm25 = None
        self.doc_term_freqs: list[Counter] = []
        self.doc_len: np.ndarray | None = None
        self.avg_doc_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.k1 = 1.5
        self.b = 0.75

    def fit(self, corpus: list[dict]) -> "SparseRetriever":
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self.doc_token_counts = {
            str(doc["doc_id"]): approximate_token_count(doc["content"])
            for doc in corpus
        }
        tokenized = [tokenize(doc["content"]) for doc in corpus]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(tokenized)
        except Exception:
            self.bm25 = None
        self.doc_term_freqs = [Counter(tokens) for tokens in tokenized]
        self.doc_len = np.asarray([len(tokens) for tokens in tokenized], dtype=np.float32)
        self.avg_doc_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        doc_freq: Counter = Counter()
        for terms in self.doc_term_freqs:
            for term in terms:
                doc_freq[term] += 1
        n_docs = len(self.doc_term_freqs)
        self.idf = {
            term: np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        return self

    def search(self, query: str, topk: int = 10) -> tuple[list[dict], float]:
        if self.doc_len is None:
            raise RuntimeError("SparseRetriever is not fitted.")
        t0 = time.perf_counter()
        doc_token_counts = getattr(self, "doc_token_counts", {})
        query_terms = tokenize(query)
        if self.bm25 is not None:
            scores = np.asarray(self.bm25.get_scores(query_terms), dtype=np.float32)
            results = _topk_from_scores(self.doc_ids, scores, topk, doc_token_counts)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return results, latency_ms
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        for idx, tf in enumerate(self.doc_term_freqs):
            doc_score = 0.0
            doc_length = float(self.doc_len[idx])
            norm = self.k1 * (1.0 - self.b + self.b * doc_length / max(self.avg_doc_len, 1e-8))
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                doc_score += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
            scores[idx] = doc_score
        results = _topk_from_scores(self.doc_ids, scores, topk, doc_token_counts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms


class DenseRetriever:
    kind = "dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_dim: int = 256,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.dense_dim = dense_dim
        self.batch_size = batch_size
        self.doc_ids: list[str] = []
        self.doc_token_counts: dict[str, int] = {}
        self.backend = "hashing"
        self.model = None
        self.doc_vectors: np.ndarray | None = None
        self.faiss_index = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["model"] = None
        if self.faiss_index is not None:
            import faiss

            state["faiss_index"] = faiss.serialize_index(self.faiss_index)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        faiss_blob = state.get("faiss_index")
        self.__dict__.update(state)
        self.model = None
        if faiss_blob is not None and isinstance(faiss_blob, (bytes, bytearray, np.ndarray)):
            import faiss

            self.faiss_index = faiss.deserialize_index(faiss_blob)

    def _load_sentence_transformer(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(self.model_name)
        return self.model

    def _fit_sentence_transformer(self, texts: list[str]) -> bool:
        try:
            import faiss

            model = self._load_sentence_transformer()
            matrix = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.doc_vectors = np.asarray(matrix, dtype=np.float32)
            dim = int(self.doc_vectors.shape[1])
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.doc_vectors)
            self.backend = "sentence-transformers"
            return True
        except Exception:
            self.model = None
            self.faiss_index = None
            return False

    def _hashed_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dense_dim, dtype=np.float32)
        for token in tokenize(text):
            vec[stable_hash(f"tok:{token}", self.dense_dim)] += 1.0
        for gram in char_ngrams(text):
            vec[stable_hash(f"chr:{gram}", self.dense_dim)] += 0.3
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit(self, corpus: list[dict]) -> "DenseRetriever":
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self.doc_token_counts = {
            str(doc["doc_id"]): approximate_token_count(doc["content"])
            for doc in corpus
        }
        texts = [doc["content"] for doc in corpus]
        if not self._fit_sentence_transformer(texts):
            matrix = np.vstack([self._hashed_vector(text) for text in texts])
            self.doc_vectors = l2_normalize(matrix.astype(np.float32))
            self.backend = "hashing"
        return self

    def _encode_query(self, query: str) -> np.ndarray:
        if self.backend == "sentence-transformers" and self.model is not None:
            vec = self.model.encode(
                [query],
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(vec, dtype=np.float32)[0]
        if self.backend == "sentence-transformers":
            model = self._load_sentence_transformer()
            vec = model.encode(
                [query],
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(vec, dtype=np.float32)[0]
        return self._hashed_vector(query)

    def search(self, query: str, topk: int = 10) -> tuple[list[dict], float]:
        if self.doc_vectors is None:
            raise RuntimeError("DenseRetriever is not fitted.")
        t0 = time.perf_counter()
        doc_token_counts = getattr(self, "doc_token_counts", {})
        qvec = self._encode_query(query)
        if self.backend == "sentence-transformers" and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(qvec[np.newaxis, :].astype(np.float32), topk)
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                if idx < 0:
                    continue
                doc_id = self.doc_ids[int(idx)]
                results.append(
                    {
                        "doc_id": doc_id,
                        "score": float(score),
                        "rank": rank,
                        "token_count": int(doc_token_counts.get(str(doc_id), 0)),
                    }
                )
        else:
            scores = self.doc_vectors @ qvec
            results = _topk_from_scores(self.doc_ids, scores, topk, doc_token_counts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms


class HybridRetriever:
    kind = "hybrid"

    def __init__(self, sparse: SparseRetriever, dense: DenseRetriever, rrf_k: int = 60) -> None:
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, topk: int = 10) -> tuple[list[dict], float]:
        sparse_results, sparse_ms = self.sparse.search(query, topk=topk * 3)
        dense_results, dense_ms = self.dense.search(query, topk=topk * 3)
        fused: dict[str, float] = defaultdict(float)
        token_counts: dict[str, int] = {}
        for results in (sparse_results, dense_results):
            for item in results:
                doc_id = str(item["doc_id"])
                fused[doc_id] += 1.0 / (self.rrf_k + item["rank"])
                if "token_count" in item:
                    token_counts[doc_id] = int(item["token_count"])
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topk]
        output = [
            {
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
                "token_count": int(token_counts.get(str(doc_id), 0)),
            }
            for rank, (doc_id, score) in enumerate(ordered, start=1)
        ]
        return output, sparse_ms + dense_ms


def save_pickle(path: str | Path, obj) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)
