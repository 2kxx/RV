#!/usr/bin/env bash
# scripts/v_arr_mvp/run_dim_pipeline.sh
# ----------------------------------------------------------------------------
# Generate the v2_prompt_pair Auto-Rubric for ONE EvalVerse second-level
# dimension. Skips the test-set evaluation pass — this script's only job is
# to land a rubric text file in
#     rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt
# so that the per-dimension reward model can reuse it later.
#
# This is intentionally lighter than scripts/v_arr_mvp/run_mvp_pipeline.sh
# because we have no per-dimension shared_test_pointwise yet (would need
# ~30 hand-translated test prompts per dim; we'll add that in a follow-up).
#
# Required env / args:
#   --dimension <name>   one of the 17 EvalVerse dim slugs (NOT visual_quality)
#
# Optional env (with sensible defaults):
#   DATA_DIR             default: data/v_arr_mvp/<dim>_mvp
#   CONFIG_PATH          default: rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml
#   RUBRICS_FILE         default: rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt
#   ARTIFACT_DIR         default: runs/v_arr_mvp_dim/<dim>/v2_prompt_pair
#   BASE_URL             default: http://v2.open.venus.oa.com/llmproxy
#   CONCURRENCY          default: 4
#
# Reuse policy: if RUBRICS_FILE already exists and is non-empty, the script
# reuses it verbatim and skips the Gemini call entirely (mirrors the B3
# behaviour added to scripts/v_arr_mvp/run_mvp_pipeline.sh).
# ----------------------------------------------------------------------------
set -euo pipefail

# ---------- argparse-lite ---------------------------------------------------
DIM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dimension) DIM="$2"; shift 2 ;;
    --dimension=*) DIM="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,28p' "$0" | sed 's/^#\? \?//'
      exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${DIM}" ]]; then
  echo "[error] --dimension <name> is required" >&2
  echo "        valid: $(python -c 'import sys;sys.path.insert(0,\"scripts/v_arr_mvp\");from dimension_registry import list_dimension_names;print(\",\".join(n for n in list_dimension_names() if n!=\"visual_quality\"))')" >&2
  exit 2
fi

# Reject visual_quality — that one is the RewardVerse MVP baseline experiment and
# already has its own pipeline (run_all_variants.sh / run_mvp_pipeline.sh).
if [[ "${DIM}" == "visual_quality" ]]; then
  echo "[error] dimension 'visual_quality' is handled by run_all_variants.sh, not this script." >&2
  exit 2
fi

# Validate dim against the registry early so we fail fast on typos.
python - "$DIM" <<'PYEOF'
import sys
sys.path.insert(0, 'scripts/v_arr_mvp')
from dimension_registry import DIMENSIONS_BY_NAME
slug = sys.argv[1].lower().strip()
if slug not in DIMENSIONS_BY_NAME:
    print(f"[error] unknown dimension {slug!r}", file=sys.stderr)
    sys.exit(2)
PYEOF

DATA_DIR="${DATA_DIR:-data/v_arr_mvp/${DIM}_mvp}"
CONFIG_PATH="${CONFIG_PATH:-rubric_pipeline/config/v_arr_mvp_${DIM}_t2v.yaml}"
RUBRICS_FILE="${RUBRICS_FILE:-rubric_pipeline/rubrics/v_arr_mvp_${DIM}_v2.txt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-runs/v_arr_mvp_dim/${DIM}/v2_prompt_pair}"
BASE_URL="${BASE_URL:-http://v2.open.venus.oa.com/llmproxy}"
CONCURRENCY="${CONCURRENCY:-4}"

mkdir -p "${ARTIFACT_DIR}" "$(dirname "${RUBRICS_FILE}")"

# ---------- 1. Build the v2_prompt_pair seed if missing ---------------------
SEED_DATASET="${DATA_DIR}/v2_prompt_pair.json"
if [[ ! -s "${SEED_DATASET}" ]]; then
  echo "[step 1/2] building seed dataset for dim=${DIM}"
  # Run in cache-only translate mode if no API key — the 17-dim cache is
  # already pre-populated by scripts/v_arr_mvp/_inject_dim17_translations.py.
  python scripts/v_arr_mvp/build_dim_seed_v2.py \
    --dimension "${DIM}" \
    --output_dir "${DATA_DIR}"
else
  echo "[step 1/2] reusing existing seed: ${SEED_DATASET}"
fi

# ---------- 2. Generate (or reuse) the rubric -------------------------------
echo "[step 2/2] running Auto-Rubric on ${SEED_DATASET}"
echo "  CONFIG       = ${CONFIG_PATH}"
echo "  RUBRICS_FILE = ${RUBRICS_FILE}"
echo "  ARTIFACT_DIR = ${ARTIFACT_DIR}"

if [[ -s "${RUBRICS_FILE}" ]]; then
  echo "[info] reusing existing rubric file ${RUBRICS_FILE} (skipping generation)."
  echo "       Delete it first if you want a fresh rubric."
  exit 0
fi

# We deliberately do NOT pass --test_dataset here. judger.py without
# --test_dataset only generates the rubric (RQ1 path). This makes the script
# fast (~1-2 min) since we skip the 30-clip evaluation pass.
python judger.py \
  --config_path "${CONFIG_PATH}" \
  --seed_dataset "${SEED_DATASET}" \
  --base_url "${BASE_URL}" \
  --concurrency_limit "${CONCURRENCY}" \
  --rubrics_out "${RUBRICS_FILE}" \
  2>&1 | tee "${ARTIFACT_DIR}/rubric_gen.log"

if [[ ! -s "${RUBRICS_FILE}" ]]; then
  echo "[error] ${RUBRICS_FILE} is missing or empty after rubric generation." >&2
  echo "        Inspect ${ARTIFACT_DIR}/rubric_gen.log." >&2
  exit 3
fi

echo
echo "[done] dim=${DIM}: rubric -> ${RUBRICS_FILE}"
wc -l "${RUBRICS_FILE}" || true
