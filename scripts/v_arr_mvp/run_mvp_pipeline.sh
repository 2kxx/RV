#!/usr/bin/env bash
# End-to-end RewardVerse MVP driver. Each step is idempotent and can be re-run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-rubric_pipeline/config/v_arr_mvp_t2v.yaml}"
BASE_URL="${BASE_URL:-http://v2.open.venus.oa.com/llmproxy}"
MODEL="${MODEL:-gemini-3.1-pro}"
CONCURRENCY="${CONCURRENCY:-2}"
SEED_DATASET="${SEED_DATASET:-examples/v_arr_mvp_seed_pairwise.json}"
TEST_DATASET="${TEST_DATASET:-examples/v_arr_mvp_test_pairwise.json}"
RUBRICS_FILE="${RUBRICS_FILE:-rubric_pipeline/rubrics/v_arr_mvp_t2v_v1.txt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-runs/v_arr_mvp}"

mkdir -p "${ARTIFACT_DIR}" rubric_pipeline/rubrics

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is required (Venus llmproxy token)." >&2
  exit 2
fi

# Tencent venus llmproxy (the OpenAI-compatible gateway used in this MVP) only
# accepts static images (JPEG/PNG/WebP/GIF). It rejects mp4 URLs ("下载图片异常")
# and any non-image bytes ("不支持的图片格式"). We therefore default to
# uniform_frames, which extracts N JPEG/PNG frames per clip locally via decord
# and ships them as image_url/base64. decord cannot read remote URLs, so videos
# must be downloaded first (see scripts/v_arr_mvp/prefetch_videos.py and the
# DATA_DIR=...visual_quality_mvp_local convention used by run_all_variants.sh).
# Override by exporting ARR_VIDEO_INPUT_STRATEGY=native_video if you really want
# to send the URL through (only works for backends that accept video URLs).
if [[ -z "${ARR_VIDEO_INPUT_STRATEGY:-}" ]]; then
  export ARR_VIDEO_INPUT_STRATEGY="uniform_frames"
  echo "[info] defaulting ARR_VIDEO_INPUT_STRATEGY=uniform_frames"
fi
if [[ -z "${ARR_VIDEO_FRAME_COUNT:-}" ]]; then
  export ARR_VIDEO_FRAME_COUNT="4"
fi

echo "[1/5] Auto-Rubric generation + pairwise eval (RQ1)"
# B3 (2026-05): if a non-empty rubric text file already exists at RUBRICS_FILE
# we REUSE it verbatim (skip rubric generation) and only run the eval pass.
# This makes prompt-template / grader-aggregation experiments cheap, since
# we don't need to re-generate rubrics every time we tweak the scoring layer.
# To force re-generation, delete RUBRICS_FILE before running this script.
if [[ -s "${RUBRICS_FILE}" ]]; then
  echo "[info] reusing existing rubric file ${RUBRICS_FILE} (skipping generation)"
  python judger.py \
    --config_path "${CONFIG_PATH}" \
    --rubrics_file "${RUBRICS_FILE}" \
    --test_dataset "${TEST_DATASET}" \
    --base_url "${BASE_URL}" \
    --concurrency_limit "${CONCURRENCY}" \
    --metrics_out "${ARTIFACT_DIR}/rubric_judge.json" \
    2>&1 | tee "${ARTIFACT_DIR}/rubric_judge.log"
else
  python judger.py \
    --config_path "${CONFIG_PATH}" \
    --seed_dataset "${SEED_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --base_url "${BASE_URL}" \
    --concurrency_limit "${CONCURRENCY}" \
    --rubrics_out "${RUBRICS_FILE}" \
    --metrics_out "${ARTIFACT_DIR}/rubric_judge.json" \
    2>&1 | tee "${ARTIFACT_DIR}/rubric_judge.log"
fi

if [[ ! -s "${RUBRICS_FILE}" ]]; then
  echo "[warn] ${RUBRICS_FILE} is missing or empty after rubric generation."
  echo "       RQ2/RQ3 will be skipped. Inspect ${ARTIFACT_DIR}/rubric_judge.log."
fi

echo "[2/5] Zero-shot baseline (RQ4)"
if [[ "${SKIP_BASELINE:-0}" == "1" ]]; then
  echo "[info] SKIP_BASELINE=1; assuming caller has already produced ${ARTIFACT_DIR}/zero_shot.json (e.g. shared baseline computed once for all variants)."
else
# Auto-detect dataset format: pairwise (response is list with 2 items + label_rank)
# vs pointwise (response is single string + label_score). Each variant feeds the
# matching baseline so that v1/v3 (pointwise) and v2/v4 (pairwise) all produce a
# usable zero_shot.json.
ZERO_SHOT_MODE="$(python - "${TEST_DATASET}" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if not isinstance(data, list) or not data:
    print("pairwise"); raise SystemExit
first = data[0]
resp = first.get("response")
if isinstance(resp, list) and len(resp) >= 2:
    print("pairwise")
elif "label_rank" in first:
    print("pairwise")
elif "label_score" in first:
    print("pointwise")
else:
    print("pairwise")
PYEOF
)"
echo "[info] zero-shot mode: ${ZERO_SHOT_MODE}"

if [[ "${ZERO_SHOT_MODE}" == "pointwise" ]]; then
  python scripts/v_arr_mvp/pointwise_zero_shot_judge.py \
    --test_dataset "${TEST_DATASET}" \
    --base_url "${BASE_URL}" \
    --model "${MODEL}" \
    --concurrency_limit "${CONCURRENCY}" \
    --output_json "${ARTIFACT_DIR}/zero_shot.json"
else
  python scripts/v_arr_mvp/zero_shot_judge.py \
    --test_dataset "${TEST_DATASET}" \
    --base_url "${BASE_URL}" \
    --model "${MODEL}" \
    --concurrency_limit "${CONCURRENCY}" \
    --output_json "${ARTIFACT_DIR}/zero_shot.json"
fi
fi  # SKIP_BASELINE

if [[ -f "${RUBRICS_FILE}" ]]; then
  echo "[3/5] Positional bias (RQ3)"
  if [[ "${SKIP_POSITIONAL:-0}" == "1" ]]; then
    echo "[info] SKIP_POSITIONAL=1; positional bias is pairwise-only, skipping (test set is pointwise)."
  else
    python scripts/v_arr_mvp/positional_bias.py \
      --config_path "${CONFIG_PATH}" \
      --rubrics_file "${RUBRICS_FILE}" \
      --test_dataset "${TEST_DATASET}" \
      --base_url "${BASE_URL}" \
      --concurrency_limit "${CONCURRENCY}" \
      --output_json "${ARTIFACT_DIR}/positional_bias.json"
  fi

  echo "[4/5] Rubric category classification (RQ2)"
  python scripts/v_arr_mvp/classify_rubric_categories.py \
    --rubrics_path "${RUBRICS_FILE}" \
    --base_url "${BASE_URL}" \
    --model "${MODEL}" \
    --output_json "${ARTIFACT_DIR}/rubric_categories.json"
else
  echo "[3-4/5] Skipping RQ2/RQ3 because ${RUBRICS_FILE} is missing."
fi

echo "[5/5] Build MVP decision report"
ZERO_SHOT_JSON_PATH="${ZERO_SHOT_JSON_OVERRIDE:-${ARTIFACT_DIR}/zero_shot.json}"
python scripts/v_arr_mvp/build_mvp_report.py \
  --rubric_judge_json "${ARTIFACT_DIR}/rubric_judge.json" \
  --zero_shot_json "${ZERO_SHOT_JSON_PATH}" \
  --positional_bias_json "${ARTIFACT_DIR}/positional_bias.json" \
  --rubric_categories_json "${ARTIFACT_DIR}/rubric_categories.json" \
  --output "${ARTIFACT_DIR}/mvp_report.md"

echo "Done. Inspect ${ARTIFACT_DIR}/mvp_report.md"
