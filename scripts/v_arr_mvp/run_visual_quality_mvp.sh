#!/usr/bin/env bash
# End-to-end driver for the Visual Quality MVP:
#  1) Learn rubric(s). DEFAULT = ONLY the `visual_quality` dimension; this is
#     the cheapest path for the MVP. Set LEARN_DIMENSIONS=all to learn rubrics
#     for every dimension present in the seed (e.g. the 18 EvalVerse
#     second-level dims), or LEARN_DIMENSIONS=visual_quality,chromaticity to
#     pick an explicit subset.
#  2) Run the Visual Quality MVP: PLCC + SRCC over a small held-out sample
#     for several video input strategies and emit a comparison report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml}"
BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
CONCURRENCY="${CONCURRENCY:-2}"

LEARNING_SEED="${LEARNING_SEED:-examples/evalverse_raw_pointwise.json}"
TEST_DATASET="${TEST_DATASET:-examples/evalverse_visual_quality_seed.json}"
RUBRICS_DIR="${RUBRICS_DIR:-rubric_pipeline/rubrics}"
ARTIFACT_DIR="${ARTIFACT_DIR:-runs/v_arr_mvp/visual_quality}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-30}"
STRATEGIES="${STRATEGIES:-native_video,frames_4,frames_8,frames_16}"
SEED="${SEED:-0}"
# MVP default: learn only the visual_quality rubric. Override examples:
#   LEARN_DIMENSIONS=all                              -> every dim in the seed
#   LEARN_DIMENSIONS=visual_quality,chromaticity      -> two-dim subset
LEARN_DIMENSIONS="${LEARN_DIMENSIONS:-visual_quality}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY must be set (used by the OpenAI-compatible client)." >&2
  exit 2
fi

mkdir -p "${RUBRICS_DIR}" "${ARTIFACT_DIR}"

# Build the optional dimensions allowlist arg. "all" means "no filter".
LEARN_DIM_ARG=()
if [[ "${LEARN_DIMENSIONS}" != "all" && -n "${LEARN_DIMENSIONS}" ]]; then
  LEARN_DIM_ARG=(--dimensions "${LEARN_DIMENSIONS}")
  echo "[1/2] Rubric learning (subset: ${LEARN_DIMENSIONS})"
else
  echo "[1/2] Rubric learning (ALL dimensions in seed -- e.g. the 18 EvalVerse second-level dims)"
fi

python scripts/v_arr_mvp/generate_per_dim_rubrics.py \
  --seed_dataset "${LEARNING_SEED}" \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${RUBRICS_DIR}" \
  --base_url "${BASE_URL}" \
  "${LEARN_DIM_ARG[@]}" \
  2>&1 | tee "${ARTIFACT_DIR}/rubric_learning.log"

VQ_RUBRIC="${RUBRICS_DIR}/visual_quality.txt"
if [[ ! -s "${VQ_RUBRIC}" ]]; then
  echo "[error] No rubric file produced at ${VQ_RUBRIC}." >&2
  echo "        Inspect ${ARTIFACT_DIR}/rubric_learning.log and ${RUBRICS_DIR}/_generation_summary.json" >&2
  exit 3
fi

echo "[2/2] Visual Quality MVP — input strategy comparison via PLCC + SRCC"
python scripts/v_arr_mvp/visual_quality_mvp_evaluation.py \
  --test_dataset "${TEST_DATASET}" \
  --config_path "${CONFIG_PATH}" \
  --rubrics_file "${VQ_RUBRIC}" \
  --dimension visual_quality \
  --strategies "${STRATEGIES}" \
  --limit "${SAMPLE_LIMIT}" \
  --seed "${SEED}" \
  --base_url "${BASE_URL}" \
  --concurrency_limit "${CONCURRENCY}" \
  --output_dir "${ARTIFACT_DIR}"

echo "Done. Inspect ${ARTIFACT_DIR}/visual_quality_strategy_comparison.md"
