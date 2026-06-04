#!/usr/bin/env bash
# scripts/v_arr_mvp/run_dim_all.sh
# ----------------------------------------------------------------------------
# Sequentially run scripts/v_arr_mvp/run_dim_pipeline.sh for all 17 EvalVerse
# second-level dimensions (everything except visual_quality, which is the
# RewardVerse MVP baseline and already has its own pipeline).
#
# Each dim:
#   1. builds data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json (cache-only translate)
#   2. calls Gemini once via judger.py to produce
#      rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt
#
# Required env (must be exported BEFORE running this script):
#   OPENAI_API_KEY     your venus llmproxy token
#   VENUS_SECRET_ID    venus storage credential (for native_video uploads)
#   VENUS_SECRET_KEY   venus storage credential
#
# Optional env:
#   CONCURRENCY        default: 4   (per-dim Gemini concurrency)
#   SKIP_EXISTING      default: 1   (skip dims whose rubric file already exists)
#   ARR_VIDEO_BLOCK_TYPE  default: gemini_multimodal_url
#                                    (matches scripts/v_arr_mvp/run_all_variants.sh)
#
# Recommended launcher (creates the runs dir BEFORE the shell redirect):
#   mkdir -p runs/v_arr_mvp_dim
#   nohup bash scripts/v_arr_mvp/run_dim_all.sh \
#     > runs/v_arr_mvp_dim/_runall.out 2>&1 & disown
# ----------------------------------------------------------------------------
set -uo pipefail

# Order: cheap-and-clean dimensions first, sparse dims (logic/rhythm/soundscape)
# last so that early failures don't block the easy ones.
DIMENSIONS=(
  pacing
  character
  scene
  consistency
  composition
  lens
  chromaticity
  materiality
  lighting
  grounding
  progression
  action
  expression
  rhythm
  vocal
  logic
  soundscape
)

CONCURRENCY="${CONCURRENCY:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
export ARR_VIDEO_BLOCK_TYPE="${ARR_VIDEO_BLOCK_TYPE:-gemini_multimodal_url}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is not exported." >&2
  echo "        export OPENAI_API_KEY=<venus_token> first." >&2
  exit 2
fi

OUT_ROOT="runs/v_arr_mvp_dim"
mkdir -p "${OUT_ROOT}"

SUMMARY="${OUT_ROOT}/all_dims_summary.md"
{
  echo "# RewardVerse MVP — 17-dimension v2_prompt_pair Auto-Rubric run"
  echo
  echo "Each row: rubric generated from 30 (or fewer if source data is sparse)"
  echo "v2-style listwise pairs for ONE EvalVerse second-level dimension."
  echo
  echo "| # | Dimension | Pairs | Rubric file | Lines |"
  echo "|---|---|---|---|---|"
} > "${SUMMARY}"

idx=0
for dim in "${DIMENSIONS[@]}"; do
  idx=$((idx + 1))
  echo "============================================================"
  echo " [$idx/${#DIMENSIONS[@]}] dim = ${dim}"
  echo "============================================================"

  RUBRIC_FILE="rubric_pipeline/rubrics/v_arr_mvp_${dim}_v2.txt"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${RUBRIC_FILE}" ]]; then
    echo "[skip] ${RUBRIC_FILE} already exists; not regenerating."
  else
    DATA_DIR="data/v_arr_mvp/${dim}_mvp"
    LOG_FILE="${OUT_ROOT}/${dim}.log"
    if ! CONCURRENCY="${CONCURRENCY}" \
         bash scripts/v_arr_mvp/run_dim_pipeline.sh \
         --dimension "${dim}" \
         > "${LOG_FILE}" 2>&1; then
      echo "[FAIL] ${dim} — see ${LOG_FILE}" | tee -a "${OUT_ROOT}/_failures.log"
      continue
    fi
  fi

  # Append to summary
  PAIRS=0
  if [[ -s "data/v_arr_mvp/${dim}_mvp/v2_prompt_pair.json" ]]; then
    PAIRS=$(python -c "import json,sys;print(len(json.load(open('data/v_arr_mvp/${dim}_mvp/v2_prompt_pair.json'))))" 2>/dev/null || echo "?")
  fi
  LINES="?"
  if [[ -s "${RUBRIC_FILE}" ]]; then
    LINES=$(wc -l < "${RUBRIC_FILE}" | tr -d ' ')
  fi
  echo "| ${idx} | \`${dim}\` | ${PAIRS} | \`${RUBRIC_FILE}\` | ${LINES} |" >> "${SUMMARY}"
done

echo
echo "============================================================"
echo " all done. summary -> ${SUMMARY}"
echo "============================================================"
cat "${SUMMARY}"
