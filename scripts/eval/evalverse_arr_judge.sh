#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# EvalVerse track selector (t2v | r2v | multishot | video_sound)
TRACK="${TRACK:-t2v}"
case "${TRACK}" in
  t2v) DEFAULT_CONFIG="rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml" ;;
  r2v) DEFAULT_CONFIG="rubric_pipeline/config/qwen3vl_8B_instruct_r2v.yaml" ;;
  multishot) DEFAULT_CONFIG="rubric_pipeline/config/qwen3vl_8B_instruct_multishot.yaml" ;;
  video_sound) DEFAULT_CONFIG="rubric_pipeline/config/qwen3vl_8B_instruct_video_sound.yaml" ;;
  *) echo "Unknown TRACK=${TRACK}"; exit 2 ;;
esac

CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG}}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
CONCURRENCY="${CONCURRENCY:-2}"
RAW_INPUT="${RAW_INPUT:-examples/evalverse_raw_pairwise.json}"
SEED_DATASET="${SEED_DATASET:-examples/seed_t2v_pairwise.json}"
TEST_DATASET="${TEST_DATASET:-examples/test_t2v_pairwise.json}"
RUBRICS_FILE="${RUBRICS_FILE:-rubric_pipeline/rubrics/evalverse_${TRACK}_v1.txt}"

mkdir -p rubric_pipeline/rubrics

echo "[1/3] Convert EvalVerse raw JSON (if present)..."
if [[ -f "${RAW_INPUT}" ]]; then
  python scripts/eval/convert_evalverse_dataset.py \
    --input "${RAW_INPUT}" \
    --output "${SEED_DATASET}"
fi

echo "[2/3] Generate rubrics from seed and evaluate on test set..."
python judger.py \
  --config_path "${CONFIG_PATH}" \
  --seed_dataset "${SEED_DATASET}" \
  --test_dataset "${TEST_DATASET}" \
  --base_url "${BASE_URL}" \
  --concurrency_limit "${CONCURRENCY}" \
  2>&1 | tee /tmp/evalverse_arr_judge.log

echo "[3/3] Re-run evaluation with saved rubric file (create placeholder if generation only)..."
if [[ ! -f "${RUBRICS_FILE}" ]]; then
  echo "Note: rubrics were printed during generation. Save them to ${RUBRICS_FILE} for reuse."
  exit 0
fi

python judger.py \
  --config_path "${CONFIG_PATH}" \
  --rubrics_file "${RUBRICS_FILE}" \
  --test_dataset "${TEST_DATASET}" \
  --base_url "${BASE_URL}" \
  --concurrency_limit "${CONCURRENCY}"
