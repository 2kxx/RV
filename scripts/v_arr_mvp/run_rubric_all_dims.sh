#!/usr/bin/env bash
# scripts/v_arr_mvp/run_rubric_all_dims.sh
# ----------------------------------------------------------------------------
# Run scripts/v_arr_mvp/pointwise_rubric_judge.py sequentially over ALL 18
# EvalVerse L2 dimensions. Uses the pre-learned per-dimension rubrics
# under rubric_pipeline/rubrics/ and the matching per-dimension yamls
# under rubric_pipeline/config/.
#
# Output: runs/v_arr_mvp_dim/<dim>/pointwise.json (baseline-schema-compatible)
#         runs/v_arr_mvp_dim/_runall_summary.tsv  (PLCC/SRCC/MAE_zs per dim)
#
# Required env:
#   OPENAI_API_KEY   venus llmproxy token (consumed by judger via OpenAI client)
#
# Optional env:
#   ARR_OUT_DIR      default: runs/v_arr_mvp_dim
#   CONCURRENCY      default: 4
#   SKIP_EXISTING    default: 1
#   MAX_RETRIES      default: 2
#   ONLY_DIMS        optional, e.g. ONLY_DIMS=rhythm,action
#   LIMIT            optional, --limit N for smoke-test
#
# Usage::
#
#     bash scripts/v_arr_mvp/run_rubric_all_dims.sh
#
# Background launcher (recommended — Gemini API takes a while):
#   mkdir -p runs/v_arr_mvp_dim
#   nohup bash scripts/v_arr_mvp/run_rubric_all_dims.sh \
#     > runs/v_arr_mvp_dim/_runall.out 2>&1 & disown
# ----------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

OUT_DIR="${ARR_OUT_DIR:-runs/v_arr_mvp_dim}"
CONCURRENCY="${CONCURRENCY:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
ONLY_DIMS="${ONLY_DIMS:-}"
LIMIT="${LIMIT:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[fatal] OPENAI_API_KEY is not exported." >&2
  echo "        export OPENAI_API_KEY=<venus_token> before running." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

# All 18 EvalVerse L2 dims, ordered by L1 cluster (matches the convention
# used in run_unifiedreward_all_dims.sh). visual_quality uses the legacy
# shared_test_pointwise.json filename; everything else uses the
# canonical test_pointwise_<dim>.json output of build_dim_seed_v2.py.
DIMS_RAW=(
  # Visual Concept Design
  "character:test_pointwise_character.json"
  "scene:test_pointwise_scene.json"
  # Acting
  "consistency:test_pointwise_consistency.json"
  "action:test_pointwise_action.json"
  "expression:test_pointwise_expression.json"
  # Cinematography
  "composition:test_pointwise_composition.json"
  "pacing:test_pointwise_pacing.json"
  "lens:test_pointwise_lens.json"
  # Aesthetics
  "visual_quality:shared_test_pointwise.json"
  "chromaticity:test_pointwise_chromaticity.json"
  "lighting:test_pointwise_lighting.json"
  "materiality:test_pointwise_materiality.json"
  # Affectivity
  "grounding:test_pointwise_grounding.json"
  "progression:test_pointwise_progression.json"
  # Multi-Shot
  "logic:test_pointwise_logic.json"
  "rhythm:test_pointwise_rhythm.json"
  # Sound Design
  "vocal:test_pointwise_vocal.json"
  "soundscape:test_pointwise_soundscape.json"
)

if [[ -n "${ONLY_DIMS}" ]]; then
  IFS=',' read -ra _ONLY <<<"${ONLY_DIMS}"
  declare -A _ALLOW=()
  for d in "${_ONLY[@]}"; do _ALLOW["${d}"]=1; done
  _FILTERED=()
  for row in "${DIMS_RAW[@]}"; do
    dim="${row%%:*}"
    if [[ -n "${_ALLOW[${dim}]:-}" ]]; then
      _FILTERED+=("${row}")
    fi
  done
  DIMS_RAW=("${_FILTERED[@]}")
fi

if [[ ${#DIMS_RAW[@]} -eq 0 ]]; then
  echo "[fatal] no dimensions to run (ONLY_DIMS filtered everything out?)" >&2
  exit 2
fi

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

T_START="$(date +%s)"
TOTAL=${#DIMS_RAW[@]}
N_OK=0
N_SKIP=0
N_FAIL=0
SUMMARY_LOG="${OUT_DIR}/_runall_summary.tsv"
: >"${SUMMARY_LOG}"
echo -e "dim\tstatus\tplcc\tsrcc\tmae_zs\tn\telapsed_sec" >>"${SUMMARY_LOG}"

echo "============================================================"
echo "[runall] Auto-Rubric over ${TOTAL} dims  concurrency=${CONCURRENCY}"
echo "[runall] out_dir=${OUT_DIR}  skip_existing=${SKIP_EXISTING}  max_retries=${MAX_RETRIES}"
echo "[runall] start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for i in "${!DIMS_RAW[@]}"; do
  row="${DIMS_RAW[${i}]}"
  dim="${row%%:*}"
  fname="${row#*:}"
  idx=$((i + 1))

  dataset="data/v_arr_mvp/${dim}_mvp/${fname}"
  out_dir_dim="${OUT_DIR}/${dim}"
  out_json="${out_dir_dim}/pointwise.json"
  log_file="${out_dir_dim}/pointwise.log"
  mkdir -p "${out_dir_dim}"

  echo
  echo "------------------------------------------------------------"
  echo "[${idx}/${TOTAL}] dim=${dim}"
  echo "    dataset=${dataset}"
  echo "    output =${out_json}"
  echo "    log    =${log_file}"

  if [[ ! -f "${dataset}" ]]; then
    echo "    [skip] dataset file not found"
    N_FAIL=$((N_FAIL + 1))
    echo -e "${dim}\tMISSING_DATASET\t-\t-\t-\t0\t0" >>"${SUMMARY_LOG}"
    continue
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_json}" ]]; then
    echo "    [skip] output already exists (set SKIP_EXISTING=0 to overwrite)"
    N_SKIP=$((N_SKIP + 1))
    metrics_line="$(python -c "
import json
try:
    d=json.load(open('${out_json}'))
    m=d.get('metrics',{}) or {}
    def f(x):
        return f'{x:.4f}' if isinstance(x,(int,float)) else '-'
    print('\t'.join([f(m.get('plcc')),f(m.get('srcc')),f(m.get('mae_zs')),str(m.get('n','-'))]))
except Exception:
    print('-\t-\t-\t-')
" 2>/dev/null)"
    echo -e "${dim}\tSKIP_EXISTING\t${metrics_line}\t0" >>"${SUMMARY_LOG}"
    continue
  fi

  attempt=0
  rc=1
  t_dim_start="$(date +%s)"
  while [[ ${attempt} -le ${MAX_RETRIES} ]]; do
    attempt=$((attempt + 1))
    echo "    [run] attempt ${attempt}/$((MAX_RETRIES + 1))  $(date '+%H:%M:%S')"
    set +e
    python scripts/v_arr_mvp/pointwise_rubric_judge.py \
        --dataset "${dataset}" \
        --output  "${out_json}" \
        --dimension "${dim}" \
        --concurrency "${CONCURRENCY}" \
        "${LIMIT_ARG[@]}" \
        >"${log_file}" 2>&1
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      break
    fi
    echo "    [warn] attempt ${attempt} failed (rc=${rc}); see ${log_file}"
    # Don't burn retries on non-transient errors.
    if grep -qE "ImportError|SyntaxError|NameError|FileNotFoundError|fatal\\] " "${log_file}"; then
      echo "    [abort] non-transient error detected; not retrying."
      break
    fi
  done
  t_dim_end="$(date +%s)"
  elapsed_dim=$((t_dim_end - t_dim_start))

  if [[ ${rc} -eq 0 && -f "${out_json}" ]]; then
    metrics_line="$(python -c "
import json
d=json.load(open('${out_json}'))
m=d.get('metrics',{}) or {}
def f(x):
    return f'{x:.4f}' if isinstance(x,(int,float)) else '-'
print('\t'.join([f(m.get('plcc')),f(m.get('srcc')),f(m.get('mae_zs')),str(m.get('n','-'))]))
" 2>/dev/null)"
    echo "    [ok]  ${elapsed_dim}s  metrics=${metrics_line}"
    N_OK=$((N_OK + 1))
    echo -e "${dim}\tOK\t${metrics_line}\t${elapsed_dim}" >>"${SUMMARY_LOG}"
  else
    echo "    [fail] gave up after ${attempt} attempts (rc=${rc}, ${elapsed_dim}s)"
    N_FAIL=$((N_FAIL + 1))
    echo -e "${dim}\tFAIL_rc${rc}\t-\t-\t-\t0\t${elapsed_dim}" >>"${SUMMARY_LOG}"
  fi
done

T_END="$(date +%s)"
TOTAL_ELAPSED=$((T_END - T_START))
HH=$((TOTAL_ELAPSED / 3600))
MM=$(((TOTAL_ELAPSED % 3600) / 60))
SS=$((TOTAL_ELAPSED % 60))

echo
echo "============================================================"
echo "[runall] DONE  total=${TOTAL}  ok=${N_OK}  skip=${N_SKIP}  fail=${N_FAIL}"
printf "[runall] elapsed: %02d:%02d:%02d (%ds)\n" "${HH}" "${MM}" "${SS}" "${TOTAL_ELAPSED}"
echo "[runall] summary: ${SUMMARY_LOG}"
echo "============================================================"
echo
echo "[runall] per-dim summary:"
column -t -s $'\t' "${SUMMARY_LOG}" 2>/dev/null || cat "${SUMMARY_LOG}"

if [[ ${N_FAIL} -gt 0 ]]; then
  exit 1
fi
exit 0
