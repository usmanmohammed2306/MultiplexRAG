#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Expected Python interpreter at $PYTHON" >&2
  echo "Create the virtual environment or adjust the script before running." >&2
  exit 1
fi

DATASET_DIR="$ROOT/data/scidocs"
TRAIN_QRELS="$DATASET_DIR/raw/qrels_router_train.jsonl"
TEST_QRELS="$DATASET_DIR/raw/qrels_router_test.jsonl"
OUT_DIR="$ROOT/phase4/scidocs/ablations"

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
echo "[info] Output dir: $OUT_DIR"

if [[ ! -f "$TRAIN_QRELS" || ! -f "$TEST_QRELS" ]]; then
  echo "Missing router qrels split under $DATASET_DIR/raw" >&2
  echo "Expected both qrels_router_train.jsonl and qrels_router_test.jsonl." >&2
  exit 1
fi

echo
echo "[step] Building / checking indexes"
"$PYTHON" "$ROOT/scripts/build_index.py" --dataset-dir "$DATASET_DIR"

# Ablation 1:
# Is the current best result sensitive to the confidence threshold?
run_router \
  "router_margin_basicqm_densefb_t050" \
  --label-tie-preference hybrid \
  --label-near-tie-mode dense \
  --label-near-tie-margin 0.0 \
  --use-retrieval-features \
  --retrieval-feature-groups basic query_match \
  --router-confidence-threshold 0.50 \
  --router-fallback-mode dense

# Ablation 2:
# Is dense fallback actually the right safety policy on SCIDOCS?
run_router \
  "router_margin_basicqm_hybridfb_t045" \
  --label-tie-preference hybrid \
  --label-near-tie-mode dense \
  --label-near-tie-margin 0.0 \
  --use-retrieval-features \
  --retrieval-feature-groups basic query_match \
  --router-confidence-threshold 0.45 \
  --router-fallback-mode hybrid

echo
echo "[note] The planned 'marginfilt' ablation is not run here."
echo "[note] Reason: train_router.py does not yet support filtering near-tie or ambiguous training queries."
echo "[note] Once that feature is implemented, add a third run such as:"
echo "[note]   router_marginfilt_basicqm_densefb_t045"

echo
echo "[done] SCIDOCS Phase 4 ablation experiments completed."
echo "[done] Reports are in $OUT_DIR"
