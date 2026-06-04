#!/usr/bin/env bash
# scripts/v_arr_mvp/run_v2_resample_round1.sh
# ----------------------------------------------------------------------------
# Round-1 v2 test-set resample sweep for 6 low-PLCC dimensions.
#
# Following the chromaticity success (round0 PLCC=0.328 -> v2 PLCC=0.7841 by
# only resampling the test set with broader label coverage), this driver
# applies the SAME treatment to 6 more dimensions whose round0 PLCC < 0.55.
#
# Per dimension it does:
#   1. Resample test set (rebuild_test_pointwise_chromaticity.py supports
#      all 6 dims now; output: data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>_v2.json)
#   2. Invalidate venus_upload cache for the new 30 URLs (otherwise
#      first-eval hits stale GCS attachment paths -> 28/30 fail)
#   3. Score with the existing round0 rubric (NO rubric regeneration)
#      using the existing pointwise_rubric_judge.py
#   4. Append PLCC/SRCC/MAE_zs to a summary TSV
#
# Required env:
#   OPENAI_API_KEY   venus llmproxy token
#
# Optional env:
#   ONLY_DIMS        comma-separated, e.g. ONLY_DIMS=logic,expression
#   CONCURRENCY      default 2 (chromaticity hit 1 download error at >=4)
#   SEED             default 0 (passed to rebuild_test_pointwise script)
#   TARGET_COUNT     default 30
#
# Usage:
#   bash scripts/v_arr_mvp/run_v2_resample_round1.sh
#
# Background:
#   nohup bash scripts/v_arr_mvp/run_v2_resample_round1.sh \
#     > runs/v_arr_mvp_dim/_runall_v2_resample.out 2>&1 & disown
# ----------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CONCURRENCY="${CONCURRENCY:-2}"
SEED="${SEED:-0}"
TARGET_COUNT="${TARGET_COUNT:-30}"
ONLY_DIMS="${ONLY_DIMS:-}"
OUT_DIR="${ARR_OUT_DIR:-runs/v_arr_mvp_dim}"
VENUS_CACHE="runs/v_arr_mvp/_venus_upload_cache.json"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[fatal] OPENAI_API_KEY is not exported." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
SUMMARY_TSV="${OUT_DIR}/_runall_v2_resample_summary.tsv"
echo -e "dim\tstatus\tplcc_round0\tplcc_v2\tsrcc_v2\tmae_zs_v2\tn_v2\tlabel_std_v2\telapsed_sec" >"${SUMMARY_TSV}"

# 6 P0/P1 dimensions, ordered by round0 PLCC ascending.
# (round0 PLCC quoted from runs/v_arr_mvp_dim/_runall.out, 2026-05-22)
DIMS_RAW=(
  "logic:-0.0194"
  "expression:0.3689"
  "pacing:0.5013"
  "progression:0.5266"
  "consistency:0.5361"
  "soundscape:0.5437"
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
  echo "[fatal] no dimensions to run" >&2
  exit 2
fi

T_START="$(date +%s)"
TOTAL=${#DIMS_RAW[@]}

echo "============================================================"
echo "[v2-resample] Round-1 v2 test-set sweep over ${TOTAL} dims"
echo "[v2-resample] concurrency=${CONCURRENCY}  seed=${SEED}  target=${TARGET_COUNT}"
echo "[v2-resample] start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for i in "${!DIMS_RAW[@]}"; do
  row="${DIMS_RAW[${i}]}"
  dim="${row%%:*}"
  plcc_r0="${row#*:}"
  idx=$((i + 1))

  out_dir_dim="${OUT_DIR}/${dim}"
  mkdir -p "${out_dir_dim}"

  test_v2="data/v_arr_mvp/${dim}_mvp/test_pointwise_${dim}_v2.json"
  out_json="${out_dir_dim}/pointwise_v2.json"
  rebuild_log="${out_dir_dim}/rebuild_v2.log"
  judge_log="${out_dir_dim}/pointwise_v2.log"

  echo
  echo "------------------------------------------------------------"
  echo "[${idx}/${TOTAL}] dim=${dim}  (round0 PLCC=${plcc_r0})"
  echo "    test_v2=${test_v2}"
  echo "    output =${out_json}"
  echo "------------------------------------------------------------"

  t_dim_start="$(date +%s)"

  # ---- Step 1: rebuild v2 test set ----
  echo "    [1/3] rebuild v2 test set..."
  set +e
  python scripts/v_arr_mvp/rebuild_test_pointwise_chromaticity.py \
      --dim "${dim}" \
      --target_count "${TARGET_COUNT}" \
      --seed "${SEED}" \
      --out "${test_v2}" \
      >"${rebuild_log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 0 || ! -f "${test_v2}" ]]; then
    echo "    [fail] rebuild failed (rc=${rc}); see ${rebuild_log}"
    echo -e "${dim}\tFAIL_REBUILD\t${plcc_r0}\t-\t-\t-\t-\t-\t0" >>"${SUMMARY_TSV}"
    continue
  fi
  # Print rebuild summary
  grep -E '^\[(info|ok)\]' "${rebuild_log}" | sed 's/^/        /'

  # ---- Step 2: invalidate venus cache for new 30 URLs ----
  echo "    [2/3] invalidate venus cache for new 30 URLs..."
  python - "${VENUS_CACHE}" "${test_v2}" "${dim}" "${idx}" <<'PYEOF'
import json, sys, shutil
from pathlib import Path
cache_path = Path(sys.argv[1])
test_v2_path = Path(sys.argv[2])
dim = sys.argv[3]
idx = int(sys.argv[4])
if not cache_path.exists():
    print(f"        venus cache not found at {cache_path} - skip invalidation")
else:
    backup = cache_path.with_suffix(f".json.bak_v2_{idx:02d}_{dim}")
    if not backup.exists():
        shutil.copy(cache_path, backup)
    cache = json.load(open(cache_path))
    n_before = len(cache)
    new_test = json.load(open(test_v2_path))
    new_urls = {s['response'] for s in new_test}
    removed = 0
    for u in new_urls:
        for k in (f"url://{u}", u):
            if k in cache:
                del cache[k]
                removed += 1
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"        invalidated {removed} entries  cache: {n_before} -> {len(cache)}")
PYEOF

  # ---- Step 3: score with round0 rubric on v2 test set ----
  echo "    [3/3] pointwise_rubric_judge (concurrency=${CONCURRENCY})..."
  set +e
  python scripts/v_arr_mvp/pointwise_rubric_judge.py \
      --dataset "${test_v2}" \
      --output  "${out_json}" \
      --dimension "${dim}" \
      --concurrency "${CONCURRENCY}" \
      >"${judge_log}" 2>&1
  rc=$?
  set -e
  t_dim_end="$(date +%s)"
  elapsed_dim=$((t_dim_end - t_dim_start))

  if [[ ${rc} -ne 0 || ! -f "${out_json}" ]]; then
    echo "    [fail] judge failed (rc=${rc}); see ${judge_log}"
    echo -e "${dim}\tFAIL_JUDGE\t${plcc_r0}\t-\t-\t-\t-\t-\t${elapsed_dim}" >>"${SUMMARY_TSV}"
    continue
  fi

  # Pull metrics
  metrics_line="$(python -c "
import json
d = json.load(open('${out_json}'))
m = d.get('metrics', {}) or {}
def f(x):
    return f'{x:.4f}' if isinstance(x, (int, float)) else '-'
plcc=f(m.get('plcc')); srcc=f(m.get('srcc')); mae=f(m.get('mae_zs')); n=str(m.get('n','-'))
# label std from the test file
samples = json.load(open('${test_v2}'))
labels = [s['label_score'] for s in samples]
import statistics
std = statistics.stdev(labels) if len(labels) >= 2 else 0
print(f'{plcc}\t{srcc}\t{mae}\t{n}\t{std:.3f}')
")"
  echo "    [ok]  ${elapsed_dim}s  metrics(v2): ${metrics_line}"
  echo -e "${dim}\tOK\t${plcc_r0}\t${metrics_line}\t${elapsed_dim}" >>"${SUMMARY_TSV}"
done

T_END="$(date +%s)"
TOTAL_ELAPSED=$((T_END - T_START))

echo
echo "============================================================"
echo "[v2-resample] DONE  total=${TOTAL}  elapsed=${TOTAL_ELAPSED}s"
echo "[v2-resample] summary -> ${SUMMARY_TSV}"
echo "============================================================"
echo
column -t -s $'\t' "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"
