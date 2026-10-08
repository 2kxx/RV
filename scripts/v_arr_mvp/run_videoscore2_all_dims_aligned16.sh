#!/usr/bin/env bash
# Rerun VideoScore-2 (aligned, bf16) on the SAME 16-dim hold-out the paper
# main table uses: data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json (no
# _v2 suffix, no shared_test_pointwise.json).
#
# Only the 12 dims with an aligned VideoScore-2 head are run; the other 4
# active dims (composition, lens, expression, progression) have no aligned
# head and stay "N/A" in the paper main table by design (vocal/soundscape
# are out-of-scope for vision-only backbones, also "N/A").
#
# Output:
#   runs/baselines/videoscore2_aligned16/<dim>_test.json
#   runs/baselines/videoscore2_aligned16/_runall_summary.tsv
#
# Usage::
#   bash scripts/v_arr_mvp/run_videoscore2_all_dims_aligned16.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DEVICE="${ARR_VIDEOSCORE2_DEVICE:-cuda:0}"
DTYPE="${ARR_VIDEOSCORE2_DTYPE:-bf16}"
NUM_FRAMES="${ARR_NUM_FRAMES:-48}"
OUT_DIR="${ARR_OUT_DIR:-runs/baselines/videoscore2_aligned16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
ONLY_DIMS="${ONLY_DIMS:-}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUT_DIR}"

DIMS_RAW=(
  "visual_quality:test_pointwise_visual_quality.json:visual quality"
  "chromaticity:test_pointwise_chromaticity.json:visual quality"
  "lighting:test_pointwise_lighting.json:visual quality"
  "materiality:test_pointwise_materiality.json:visual quality"
  "consistency:test_pointwise_consistency.json:temporal consistency"
  "action:test_pointwise_action.json:dynamic degree"
  "rhythm:test_pointwise_rhythm.json:temporal consistency"
  "logic:test_pointwise_logic.json:factual consistency"
  "pacing:test_pointwise_pacing.json:dynamic degree"
  "character:test_pointwise_character.json:text-to-video alignment"
  "scene:test_pointwise_scene.json:text-to-video alignment"
  "grounding:test_pointwise_grounding.json:text-to-video alignment"
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
echo -e "dim\tstatus\tplcc\tsrcc\tmae_zs\tn\ttarget_subscore\telapsed_sec" >>"${SUMMARY_LOG}"

echo "============================================================"
echo "[runall] VideoScore2 (aligned-16 test) over ${TOTAL} dims"
echo "[runall] device=${DEVICE} dtype=${DTYPE}  out_dir=${OUT_DIR}"
echo "[runall] start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for i in "${!DIMS_RAW[@]}"; do
  row="${DIMS_RAW[${i}]}"
  dim="${row%%:*}"
  rest="${row#*:}"
  fname="${rest%%:*}"
  target="${rest#*:}"
  idx=$((i + 1))

  dataset="data/v_arr_mvp/${dim}_mvp/${fname}"
  out_json="${OUT_DIR}/${dim}_test.json"
  log_file="${OUT_DIR}/${dim}_test.log"

  echo
  echo "------------------------------------------------------------"
  echo "[${idx}/${TOTAL}] dim=${dim}  expected_target=${target}"
  echo "    dataset=${dataset}"
  echo "    output =${out_json}"

  if [[ ! -f "${dataset}" ]]; then
    echo "    [skip] dataset file not found"
    N_FAIL=$((N_FAIL + 1))
    echo -e "${dim}\tMISSING_DATASET\t-\t-\t-\t0\t${target}\t0" >>"${SUMMARY_LOG}"
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
    print('\t'.join([f(m.get('plcc')),f(m.get('srcc')),f(m.get('mae_zs')),str(m.get('n','-')),str(d.get('target_subscore','-'))]))
except Exception:
    print('-\t-\t-\t-\t-')
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
    python scripts/v_arr_mvp/baseline_videoscore2.py \
        --dataset "${dataset}" \
        --output  "${out_json}" \
        --device  "${DEVICE}" \
        --dtype   "${DTYPE}" \
        --num_frames "${NUM_FRAMES}" \
        "${LIMIT_ARG[@]}" \
        >"${log_file}" 2>&1
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      break
    fi
    echo "    [warn] attempt ${attempt} failed (rc=${rc}); see ${log_file}"
    if grep -qE "ImportError|SyntaxError|NameError" "${log_file}"; then
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
print('\t'.join([f(m.get('plcc')),f(m.get('srcc')),f(m.get('mae_zs')),str(m.get('n','-')),str(d.get('target_subscore','-'))]))
" 2>/dev/null)"
    echo "    [ok]  ${elapsed_dim}s  metrics=${metrics_line}"
    N_OK=$((N_OK + 1))
    echo -e "${dim}\tOK\t${metrics_line}\t${elapsed_dim}" >>"${SUMMARY_LOG}"
  else
    echo "    [fail] gave up after ${attempt} attempts (rc=${rc}, ${elapsed_dim}s)"
    N_FAIL=$((N_FAIL + 1))
    echo -e "${dim}\tFAIL_rc${rc}\t-\t-\t-\t0\t${target}\t${elapsed_dim}" >>"${SUMMARY_LOG}"
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
