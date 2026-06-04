#!/usr/bin/env bash
# scripts/v_arr_mvp/run_videoscore_all_dims.sh
# ----------------------------------------------------------------------------
# Run scripts/v_arr_mvp/baseline_videoscore.py sequentially over the 12
# EvalVerse L2 dimensions for which VideoScore-v1.1 has a canonical
# sub-score (per BASELINE_COVERAGE in baseline_videoscore.py).
#
# The remaining 6 dims (expression / composition / lens / progression /
# vocal / soundscape) have no semantically aligned VideoScore-v1.1 head
# and are intentionally NOT run here — the SOTA table will report "—"
# in those cells. If you need a fall-back number anyway, run them
# manually with `--target_subscore raw_overall_4`.
#
# Usage::
#
#     bash scripts/v_arr_mvp/run_videoscore_all_dims.sh
#
# Optional env:
#   ARR_VIDEOSCORE_DEVICE   default: cuda:0
#   ARR_VIDEOSCORE_DTYPE    default: bf16  (matches official README torch_dtype=bfloat16)
#   ARR_NUM_FRAMES          default: 48
#   ARR_OUT_DIR             default: runs/baselines/videoscore
#   SKIP_EXISTING           default: 1   (skip dims whose output JSON exists)
#   MAX_RETRIES             default: 2   (per-dim retries on non-zero exit)
#   ONLY_DIMS               optional, comma-separated allowlist, e.g.
#                           ONLY_DIMS=rhythm,action  bash ... .sh
#   LIMIT                   optional, --limit N for smoke-testing each dim
#
# Recommended launcher (background + single combined log):
#   mkdir -p runs/baselines/videoscore
#   nohup bash scripts/v_arr_mvp/run_videoscore_all_dims.sh \
#     > runs/baselines/videoscore/_runall.out 2>&1 & disown
# ----------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DEVICE="${ARR_VIDEOSCORE_DEVICE:-cuda:0}"
DTYPE="${ARR_VIDEOSCORE_DTYPE:-bf16}"
NUM_FRAMES="${ARR_NUM_FRAMES:-48}"
OUT_DIR="${ARR_OUT_DIR:-runs/baselines/videoscore}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
ONLY_DIMS="${ONLY_DIMS:-}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# (dim, dataset_filename, expected_target_subscore) — order chosen to put
# the strong-signal Aesthetics cluster first (visual quality head, the
# best-aligned head per EvalVerse), then Acting+Multi-Shot, then Visual
# Concept Design+Affectivity. Each row's 3rd field is documentation only
# (the script lets baseline_videoscore.py auto-pick via BASELINE_COVERAGE).
# ---------------------------------------------------------------------------
DIMS_RAW=(
  # Aesthetics cluster -> visual quality
  "visual_quality:shared_test_pointwise.json:visual quality"
  "chromaticity:test_pointwise_chromaticity_v2.json:visual quality"
  "lighting:test_pointwise_lighting.json:visual quality"
  "materiality:test_pointwise_materiality.json:visual quality"
  # Acting + Multi-Shot + Cinematography:Pacing
  "consistency:test_pointwise_consistency_v2.json:temporal consistency"
  "action:test_pointwise_action.json:dynamic degree"
  "rhythm:test_pointwise_rhythm.json:temporal consistency"
  "logic:test_pointwise_logic_v2.json:factual consistency"
  "pacing:test_pointwise_pacing_v2.json:dynamic degree"
  # Visual Concept Design + Affectivity:Grounding
  "character:test_pointwise_character.json:text-to-video alignment"
  "scene:test_pointwise_scene.json:text-to-video alignment"
  "grounding:test_pointwise_grounding.json:text-to-video alignment"
)

# Filter by ONLY_DIMS allowlist if provided.
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
echo "[runall] VideoScore-v1.1 over ${TOTAL} dims  device=${DEVICE} dtype=${DTYPE}"
echo "[runall] out_dir=${OUT_DIR}  skip_existing=${SKIP_EXISTING}  max_retries=${MAX_RETRIES}"
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
  echo "    log    =${log_file}"

  if [[ ! -f "${dataset}" ]]; then
    echo "    [skip] dataset file not found"
    N_FAIL=$((N_FAIL + 1))
    echo -e "${dim}\tMISSING_DATASET\t-\t-\t-\t0\t${target}\t0" >>"${SUMMARY_LOG}"
    continue
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_json}" ]]; then
    echo "    [skip] output already exists (set SKIP_EXISTING=0 to overwrite)"
    N_SKIP=$((N_SKIP + 1))
    # Still try to scrape metrics from the existing file for the summary.
    metrics_line="$(python -c "
import json,sys
try:
    d=json.load(open('${out_json}'))
    m=d.get('metrics',{}) or {}
    def f(x):
        return f'{x:.4f}' if isinstance(x,(int,float)) else '-'
    print('\t'.join([f(m.get('plcc')),f(m.get('srcc')),f(m.get('mae_zs')),str(m.get('n','-')),str(d.get('target_subscore','-'))]))
except Exception as e:
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
    python scripts/v_arr_mvp/baseline_videoscore.py \
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
    # Retry only on transient-looking errors (CUDA OOM, decord, network);
    # don't burn retries on syntax errors.
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

# Non-zero exit if any dim failed (so CI / nohup wrappers can detect).
if [[ ${N_FAIL} -gt 0 ]]; then
  exit 1
fi
exit 0
