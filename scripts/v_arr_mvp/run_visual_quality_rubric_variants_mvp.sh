#!/usr/bin/env bash
# Visual Quality MVP — rubric generation input ablation
#
# Three steps:
#   1. Synthesize visual_quality preference pairs from a per-video score JSONL
#      (defaults to merged_output_slim_with_new_path.jsonl).
#   2. Learn 4 rubric variants (Prompt+y+ / Prompt+y++y- / y+ / y++y-).
#   3. Compare the 4 rubrics on a small held-out test set via PLCC + SRCC.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Default points at the MoS-annotated JSONL (merged_evaluate_results_with_path_part1.jsonl).
# Set INPUT_JSONL=merged_output_slim_with_new_path.jsonl to use the older corpus
# (method_rank mode, no human MoS).
INPUT_JSONL="${INPUT_JSONL:-merged_evaluate_results_with_path_part1.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml}"
BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
CONCURRENCY="${CONCURRENCY:-2}"

PAIRS_OUT="${PAIRS_OUT:-data/v_arr_mvp/visual_quality_pairs.json}"
RUBRICS_DIR="${RUBRICS_DIR:-rubric_pipeline/rubrics}"
TEST_DATASET="${TEST_DATASET:-examples/evalverse_visual_quality_seed.json}"
ARTIFACT_DIR="${ARTIFACT_DIR:-runs/v_arr_mvp/visual_quality_rubric_variants}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-30}"
STRATEGY="${STRATEGY:-native_video}"
# `mos_aesthetics_vq` mode uses a 0-1 MoS gap (default 0.10).
# `method_rank` mode uses a 0-1 win-ratio gap (default 0.20).
MIN_SCORE_GAP="${MIN_SCORE_GAP:-0.10}"
MAX_PAIRS="${MAX_PAIRS:-30}"
SYNTH_MODE="${SYNTH_MODE:-auto}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo '[error] OPENAI_API_KEY required (used by the OpenAI-compatible client).' >&2
  exit 2
fi

mkdir -p "$(dirname "${PAIRS_OUT}")" "${RUBRICS_DIR}" "${ARTIFACT_DIR}"

echo "[1/3] Synthesize visual_quality pairs from ${INPUT_JSONL} (mode=${SYNTH_MODE})"
python scripts/v_arr_mvp/synthesize_visual_quality_pairs.py \
  --input "${INPUT_JSONL}" \
  --output "${PAIRS_OUT}" \
  --mode "${SYNTH_MODE}" \
  --min_score_gap "${MIN_SCORE_GAP}" \
  --max_total_pairs "${MAX_PAIRS}" \
  2>&1 | tee "${ARTIFACT_DIR}/synthesis.log"

if [[ ! -s "${PAIRS_OUT}" ]]; then
  echo "[error] No pairs were synthesized; inspect ${ARTIFACT_DIR}/synthesis.log" >&2
  exit 3
fi

echo "[2/3] Learn 4 rubric variants from synthesized pairs"
python scripts/v_arr_mvp/learn_visual_quality_rubric_variants.py \
  --input "${PAIRS_OUT}" \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${RUBRICS_DIR}" \
  --base_url "${BASE_URL}" \
  2>&1 | tee "${ARTIFACT_DIR}/rubric_learning.log"

RUBRIC_PATHS=(
  "${RUBRICS_DIR}/visual_quality_v1_prompt_ywin.txt"
  "${RUBRICS_DIR}/visual_quality_v2_prompt_pair.txt"
  "${RUBRICS_DIR}/visual_quality_v3_ywin_only.txt"
  "${RUBRICS_DIR}/visual_quality_v4_pair_only.txt"
)
RUBRIC_FILES_CSV="$(IFS=, ; echo "${RUBRIC_PATHS[*]}")"

echo "[3/3] Compare variants by PLCC/SRCC on the test set (strategy=${STRATEGY})"
python scripts/v_arr_mvp/visual_quality_rubric_variant_compare.py \
  --test_dataset "${TEST_DATASET}" \
  --rubric_files "${RUBRIC_FILES_CSV}" \
  --config_path "${CONFIG_PATH}" \
  --strategy "${STRATEGY}" \
  --limit "${SAMPLE_LIMIT}" \
  --base_url "${BASE_URL}" \
  --concurrency_limit "${CONCURRENCY}" \
  --output_dir "${ARTIFACT_DIR}"

echo "Done. Inspect ${ARTIFACT_DIR}/visual_quality_rubric_variant_comparison.md"
