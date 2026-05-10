#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Expected Python interpreter at $PYTHON" >&2
  echo "Create the virtual environment or adjust the script before running." >&2
  exit 1
fi

DATASET_DIR="$ROOT/data/nfcorpus"
TRAIN_QRELS="$DATASET_DIR/raw/qrels_train.jsonl"
TEST_QRELS="$DATASET_DIR/raw/qrels_validation.jsonl"
OUT_DIR="$ROOT/phase4/nfcorpus/main"

mkdir -p "$OUT_DIR"

run_router() {
  local stem="$1"
  shift

  echo
  echo "[run] $stem"
  "$PYTHON" "$ROOT/scripts/train_router.py" \
    --dataset-dir "$DATASET_DIR" \
    --train-qrels "$TRAIN_QRELS" \
    --test-qrels "$TEST_QRELS" \
    --topk 10 \
    --model-out "$OUT_DIR/${stem}.pkl" \
    --report-out "$OUT_DIR/${stem}_report.json" \
    --pred-out "$OUT_DIR/${stem}_predictions.jsonl" \
    "$@"
}

echo "[info] Root      : $ROOT"
echo "[info] Dataset   : $DATASET_DIR"
echo "[info] Train qrels: $TRAIN_QRELS"
echo "[info] Test qrels : $TEST_QRELS"
echo "[info] Output dir: $OUT_DIR"

if [[ ! -f "$TRAIN_QRELS" || ! -f "$TEST_QRELS" ]]; then
  echo "Missing NFCorpus qrels split under $DATASET_DIR/raw" >&2
  echo "Expected qrels_train.jsonl and qrels_validation.jsonl." >&2
  exit 1
fi

echo
echo "[step] Building / checking indexes"
"$PYTHON" "$ROOT/scripts/build_index.py" --dataset-dir "$DATASET_DIR"

# Main experiment 1:
# Query-only router with hybrid fallback as a dataset-aware baseline.
run_router \
  "router_base_qonly_hybridfb" \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode hybrid

# Main experiment 2:
# Add basic retrieval-confidence features with the known-good hybrid fallback.
run_router \
  "router_base_basic_hybridfb" \
  --use-retrieval-features \
  --retrieval-feature-groups basic \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode hybrid

# Main experiment 3:
# Add query-match features to the basic retrieval feature block.
run_router \
  "router_base_basicqm_hybridfb_t045" \
  --use-retrieval-features \
  --retrieval-feature-groups basic query_match \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode hybrid

# Main experiment 4:
# Keep the same best-known feature recipe and test a more conservative threshold.
run_router \
  "router_base_basicqm_hybridfb_t050" \
  --use-retrieval-features \
  --retrieval-feature-groups basic query_match \
  --router-confidence-threshold 0.50 \
  --router-fallback-mode hybrid

echo
echo "[done] NFCorpus Phase 4 main experiments completed."
echo "[done] Reports are in $OUT_DIR"
