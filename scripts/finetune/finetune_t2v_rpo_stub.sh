#!/usr/bin/env bash
# Stub launcher for T2V RPO + ARR rewards. Replace SAMPLE_CMD with your T2V trainer.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export ARR_CONFIG_PATH="${ARR_CONFIG_PATH:-rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml}"
export ARR_RUBRICS_FILE="${ARR_RUBRICS_FILE:-rubric_pipeline/rubrics/evalverse_t2v_v1.txt}"

echo "T2V RPO stub: integrate fastvideo/train_rpo_t2v.py into your sampler loop."
echo "ARR config: ${ARR_CONFIG_PATH}"
echo "Rubrics:    ${ARR_RUBRICS_FILE}"
python -c "import fastvideo.train_rpo_t2v as m; print(m.__doc__)"
