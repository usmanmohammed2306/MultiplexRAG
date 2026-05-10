#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Expected Python interpreter at $PYTHON" >&2
  echo "Create the virtual environment or adjust the script before running." >&2
  exit 1
fi

DATASET_DIR="$ROOT/data/scifact"
TRAIN_QRELS="$DATASET_DIR/raw/qrels_train.jsonl"
TEST_QRELS="$DATASET_DIR/raw/qrels_test.jsonl"
OUT_DIR="$ROOT/phase4/scifact/main"

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
  echo "Missing SciFact qrels split under $DATASET_DIR/raw" >&2
  echo "Expected qrels_train.jsonl and qrels_test.jsonl." >&2
  exit 1
fi

echo
echo "[step] Building / checking indexes"
"$PYTHON" "$ROOT/scripts/build_index.py" --dataset-dir "$DATASET_DIR"

# Main experiment 1:
# Reproduce the current best integrated SciFact router.
run_router \
  "router_base_qonly_densefb_t045" \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode dense

# Main experiment 2:
# Test whether basic retrieval features help on SciFact.
run_router \
  "router_base_basic_densefb_t045" \
  --use-retrieval-features \
  --retrieval-feature-groups basic \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode dense

# Main experiment 3:
# Test the transferable Phase 4 feature recipe on SciFact.
run_router \
  "router_base_basicqm_densefb_t045" \
  --use-retrieval-features \
  --retrieval-feature-groups basic query_match \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode dense

# Main experiment 4:
# Check whether a more conservative threshold helps or hurts on SciFact.
run_router \
  "router_base_basicqm_densefb_t050" \
  --use-retrieval-features \
  --retrieval-feature-groups basic query_match \
  --router-confidence-threshold 0.50 \
  --router-fallback-mode dense

echo
echo "[done] SciFact Phase 4 main experiments completed."
echo "[done] Reports are in $OUT_DIR"
