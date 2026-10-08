#!/usr/bin/env bash
# B2 — Gemini zero-shot, NO rubric, 16 dims (vocal/soundscape excluded;
# vision-only via Venus llmproxy native-video). Same hold-out as the
# Qwen2.5-VL untrained baselines and the rubric main result, so the paper
# main table joins on dimension + label_score on all rows.
#
# Output:
#   runs/baselines/B2_zero_shot_${ARR_MODEL}/<dim>.json   (one per dim)
#   runs/baselines/B2_zero_shot_${ARR_MODEL}/_runall_summary.tsv
#
# Required env:
#   OPENAI_API_KEY    — Venus llmproxy token
# Optional env:
#   ARR_MODEL         — Gemini model id, default "gemini-3.1-pro" (paper
#                       main table uses the LATEST you set here; e.g.
#                       export ARR_MODEL=gemini-3.5-pro before invoking).
#   ARR_BASE_URL      — Venus llmproxy base, default the one in the script
#   B2_CONCURRENCY    — async fan-out, default 2 (Gemini long context;
#                       4 is safe on llmproxy if quota allows)
#
# Usage::
#
#   export OPENAI_API_KEY=...
#   export ARR_MODEL=gemini-3.5-pro          # or whichever latest variant
#   bash scripts/v_arr_mvp/run_baseline_zero_shot_gemini_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${ARR_MODEL:-gemini-3.1-pro}"
CONC="${B2_CONCURRENCY:-2}"

# Sanitize the model name for use as a directory tag (replace "/" and ":").
MODEL_TAG="${MODEL//\//_}"
MODEL_TAG="${MODEL_TAG//:/_}"

OUT_DIR="$ROOT/runs/baselines/B2_zero_shot_${MODEL_TAG}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# 16 active dims (paper §3 scope: vision-only backbone, audio dims excluded).
DIMS=(
  visual_quality consistency grounding character action
  lens lighting composition scene chromaticity
  expression materiality logic pacing progression rhythm
)

# Resolve the per-dim test_pointwise file for each dim. Mirrors
# resolve_test_pointwise_path in eval_pointwise.py:
#   1. test_pointwise_<dim>_v2.json   (if present)
#   2. test_pointwise_<dim>.json
#   3. shared_test_pointwise.json     (visual_quality fallback)
resolve_dataset() {
  local dim="$1"
  local d="$ROOT/data/v_arr_mvp/${dim}_mvp"
  for cand in \
      "$d/test_pointwise_${dim}_v2.json" \
      "$d/test_pointwise_${dim}.json"; do
    if [ -f "$cand" ]; then echo "$cand"; return 0; fi
  done
  if [ "$dim" = "visual_quality" ] && [ -f "$d/shared_test_pointwise.json" ]; then
    echo "$d/shared_test_pointwise.json"; return 0;
  fi
  return 1
}

echo ">>> B2 zero-shot (no rubric)  model=$MODEL  out=$OUT_DIR"

for dim in "${DIMS[@]}"; do
  ds="$(resolve_dataset "$dim" || true)"
  if [ -z "${ds:-}" ]; then
    echo "  [skip] $dim (no test_pointwise file)"
    continue
  fi
  out_json="$OUT_DIR/${dim}.json"
  log="$LOG_DIR/${dim}.log"
  echo "  [$dim] dataset=$(realpath --relative-to="$ROOT" "$ds")  ->  $(realpath --relative-to="$ROOT" "$out_json")"
  python scripts/v_arr_mvp/baseline_zero_shot_18d.py \
      --mode B1b --dim "$dim" \
      --dataset "$ds" \
      --output  "$out_json" \
      --model   "$MODEL" \
      --concurrency "$CONC" \
      2>&1 | tee "$log" >/dev/null
done

# ----------------------------------------------------------------------
# Aggregate per-dim metrics into _runall_summary.tsv with the same column
# layout the Qwen baseline TSV uses, so refresh_baseline_table.py can
# join on the same code path.
# ----------------------------------------------------------------------
export ROOT OUT_DIR
DIMS_STR="${DIMS[*]}"
export DIMS_STR
python - <<'PY'
import json, os, sys
from pathlib import Path

ROOT = Path(os.environ["ROOT"])
OUT_DIR = Path(os.environ["OUT_DIR"])
DIMS = os.environ["DIMS_STR"].split()

rows = []
plccs, srccs = [], []
for dim in DIMS:
    fp = OUT_DIR / f"{dim}.json"
    if not fp.is_file():
        continue
    obj = json.loads(fp.read_text(encoding="utf-8"))
    m = obj.get("metrics") or {}
    rows.append((
        dim,
        m.get("plcc"), m.get("srcc"), m.get("mae"),
        m.get("n", 0),
        obj.get("total", 0) - obj.get("parseable", 0),
        obj.get("total", 0),
    ))
    if isinstance(m.get("plcc"), (int, float)): plccs.append(float(m["plcc"]))
    if isinstance(m.get("srcc"), (int, float)): srccs.append(float(m["srcc"]))

tsv = OUT_DIR / "_runall_summary.tsv"
with tsv.open("w", encoding="utf-8") as f:
    f.write("dim\tplcc\tsrcc\tmae\tn\tfailures\ttotal\n")
    for dim, plcc, srcc, mae, n, fail, tot in rows:
        def _fmt(x): return f"{float(x):.4f}" if isinstance(x, (int, float)) else "nan"
        f.write(f"{dim}\t{_fmt(plcc)}\t{_fmt(srcc)}\t{_fmt(mae)}\t{n}\t{fail}\t{tot}\n")
    if plccs:
        f.write(f"# mean over {len(plccs)} dims:\t"
                f"{sum(plccs)/len(plccs):+.4f}\t"
                f"{(sum(srccs)/len(srccs) if srccs else 0):+.4f}\n")

print(f"[runall] wrote {tsv}")
if plccs:
    print(f"[runall] mean PLCC over {len(plccs)} dims = {sum(plccs)/len(plccs):+.4f}")
PY

echo ">>> B2 done."
