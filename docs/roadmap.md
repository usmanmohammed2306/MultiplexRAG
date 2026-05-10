# MultiplexRAG Roadmap

## Phase 0: Define task and metrics

- Pick one benchmark/corpus.
- Decide metrics: Recall@k, MRR/nDCG, latency (ms), estimated token/cost.
- Create train/dev/test split with fixed random seed.

Exit criteria:
- You can run one script that prints dataset sizes and metric definitions.

## Phase 1: Single retriever baselines

- Sparse baseline: BM25.
- Dense baseline: sentence-transformers + FAISS.
- Save top-k doc IDs for each query.

Exit criteria:
- Table with BM25 vs Dense on same split and same k.

## Phase 2: Hybrid baseline

- Implement score fusion: RRF or weighted sum.
- Tune fusion weights on dev split only.

Exit criteria:
- Hybrid score >= best single retriever on dev.

## Phase 3: Router (Multiplexer)

- Build query features: length, numbers/acronyms count, OOV ratio, embedding entropy, etc.
- Start with classifier policy: predict `sparse | dense | hybrid`.
- Add confidence threshold fallback to a configurable default branch.
- Compare single-metric and multi-metric weak-label rules.

Exit criteria:
- Router improves quality-cost tradeoff vs always-dense / always-bm25 / always-hybrid.

## Phase 4: Upper bound and analysis

- Oracle upper bound: choose best retriever per query using ground truth.
- Plot per-query wins by query type.
- Analyze errors where router chooses wrong branch.

Exit criteria:
- Final report with evidence and ablations.
