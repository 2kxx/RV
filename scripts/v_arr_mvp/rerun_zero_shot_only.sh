#!/usr/bin/env bash
# Re-run ONLY the zero-shot baseline (RQ4) for all 4 RewardVerse MVP visual-quality
# variants. Useful when the rubric judge has already finished but zero-shot
# either crashed or used the wrong baseline for pointwise data.
#
# Auto-routes:
#   v1_prompt_ywin / v3_ywin_only  -> pointwise_zero_shot_judge.py (1-5 scoring)
#   v2_prompt_pair / v4_pair_only  -> zero_shot_judge.py           (rank pairwise)
#
# Usage:
#   export OPENAI_API_KEY=<venus_token>
#   export VENUS_SECRET_ID=<your_id>
#   export VENUS_SECRET_KEY=<your_key>
#   bash scripts/v_arr_mvp/rerun_zero_shot_only.sh
#
# Optional env overrides (same defaults as run_all_variants.sh):
#   MODEL                       (default: gemini-3.1-pro)
#   BASE_URL                    (default: http://v2.open.venus.oa.com/llmproxy)
#   CONCURRENCY                 (default: 2)
#   DATA_DIR                    (default: data/v_arr_mvp/visual_quality_mvp)
#   OUT_ROOT                    (default: runs/v_arr_mvp)
#   ARR_VIDEO_INPUT_STRATEGY    (default: native_video)
#   ARR_VIDEO_BLOCK_TYPE        (default: gemini_multimodal_url)
#   ONLY                        comma-list to restrict, e.g. ONLY=v1_prompt_ywin

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is required (Venus llmproxy token)." >&2
  exit 2
fi

MODEL="${MODEL:-gemini-3.1-pro}"
BASE_URL="${BASE_URL:-http://v2.open.venus.oa.com/llmproxy}"
CONCURRENCY="${CONCURRENCY:-2}"
DATA_DIR="${DATA_DIR:-data/v_arr_mvp/visual_quality_mvp}"
OUT_ROOT="${OUT_ROOT:-runs/v_arr_mvp}"

# Default to venus native-video upload path (matches run_all_variants.sh).
export ARR_VIDEO_INPUT_STRATEGY="${ARR_VIDEO_INPUT_STRATEGY:-native_video}"
export ARR_VIDEO_BLOCK_TYPE="${ARR_VIDEO_BLOCK_TYPE:-gemini_multimodal_url}"
export ARR_VENUS_MODEL="${ARR_VENUS_MODEL:-${MODEL}}"

if [[ "${ARR_VIDEO_BLOCK_TYPE}" == "gemini_multimodal_url" ]]; then
  if [[ -z "${VENUS_SECRET_ID:-}" || -z "${VENUS_SECRET_KEY:-}" ]]; then
    echo "[error] VENUS_SECRET_ID and VENUS_SECRET_KEY are required for the" >&2
    echo "        gemini_multimodal_url upload path." >&2
    exit 2
  fi
fi

# variant_id : dataset_filename : pointwise|pairwise
VARIANTS=(
  "v1_prompt_ywin:v1_prompt_ywin.json:pointwise"
  "v2_prompt_pair:v2_prompt_pair.json:pairwise"
  "v3_ywin_only:v3_ywin_only.json:pointwise"
  "v4_pair_only:v4_pair_only.json:pairwise"
)

ONLY="${ONLY:-}"
should_run() {
  local vid="$1"
  if [[ -z "${ONLY}" ]]; then return 0; fi
  IFS=',' read -ra keep <<< "${ONLY}"
  for k in "${keep[@]}"; do [[ "${k}" == "${vid}" ]] && return 0; done
  return 1
}

echo "============================================================"
echo " RewardVerse MVP : re-running zero-shot baseline only (4 variants)"
echo " MODEL        = ${MODEL}"
echo " BASE_URL     = ${BASE_URL}"
echo " STRATEGY     = ${ARR_VIDEO_INPUT_STRATEGY} / ${ARR_VIDEO_BLOCK_TYPE}"
echo " DATA_DIR     = ${DATA_DIR}"
echo " OUT_ROOT     = ${OUT_ROOT}"
echo "============================================================"

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r vid fname mode <<< "${entry}"
  should_run "${vid}" || { echo "[skip] ${vid}"; continue; }

  dataset="${DATA_DIR}/${fname}"
  artifact="${OUT_ROOT}/${vid}"
  out_json="${artifact}/zero_shot.json"
  log_file="${artifact}/zero_shot_rerun.log"

  if [[ ! -f "${dataset}" ]]; then
    echo "[error] missing dataset: ${dataset}" >&2
    continue
  fi
  mkdir -p "${artifact}"

  echo
  echo "------------------------------------------------------------"
  echo "[zs] ${vid}  mode=${mode}"
  echo "     dataset = ${dataset}"
  echo "     output  = ${out_json}"
  echo "     log     = ${log_file}"
  echo "------------------------------------------------------------"

  set +e
  if [[ "${mode}" == "pointwise" ]]; then
    python scripts/v_arr_mvp/pointwise_zero_shot_judge.py \
      --test_dataset "${dataset}" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency_limit "${CONCURRENCY}" \
      --output_json "${out_json}" 2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
  else
    python scripts/v_arr_mvp/zero_shot_judge.py \
      --test_dataset "${dataset}" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency_limit "${CONCURRENCY}" \
      --output_json "${out_json}" 2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[warn] ${vid} zero-shot exited with code ${rc}; continuing." >&2
  fi
done

# ---- Refresh the all-variants summary table -------------------------------
SUMMARY="${OUT_ROOT}/all_variants_summary.md"
python - "${OUT_ROOT}" "${SUMMARY}" <<'PYEOF'
import json, os, re, sys, pathlib

out_root = sys.argv[1]
summary_path = sys.argv[2]
entries = [
    ("v1_prompt_ywin", "v1_prompt_ywin.json", "Prompt + y+ (Exemplar)"),
    ("v2_prompt_pair", "v2_prompt_pair.json", "Prompt + y+ + y- (ARR original)"),
    ("v3_ywin_only",   "v3_ywin_only.json",   "y+ only"),
    ("v4_pair_only",   "v4_pair_only.json",   "y+ + y- (no prompt)"),
]

def safe_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def parse_rubric_judge_log(log_path):
    if not os.path.exists(log_path):
        return None, None, None
    txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    total = re.search(r"Total Tests:\s*(\d+)", txt)
    correct = re.search(r"Correct:\s*(\d+)", txt)
    acc = re.search(r"Accuracy:\s*([\d.]+)\s*%", txt)
    return (
        int(total.group(1)) if total else None,
        int(correct.group(1)) if correct else None,
        float(acc.group(1)) if acc else None,
    )

def parse_zero_shot(p):
    d = safe_load_json(p)
    if not d:
        return None, None, None, None, None
    total = d.get("total") or d.get("total_tests") or d.get("num_total")
    correct = d.get("correct") or d.get("num_correct")
    parseable = d.get("parseable")
    acc = d.get("accuracy")
    if acc is None and total and correct is not None:
        acc = correct / total * 100.0 if total else None
    elif acc is not None and acc <= 1.0:
        acc = acc * 100.0
    mode = d.get("mode") or "pairwise"
    return total, correct, parseable, acc, mode

rows = []
for vid, fname, label in entries:
    art = os.path.join(out_root, vid)
    rubric_log = os.path.join(art, "rubric_judge.log")
    pipe_log = os.path.join(art, "pipeline.log")
    total_r, correct_r, acc_r = parse_rubric_judge_log(rubric_log)
    if acc_r is None:
        total_r, correct_r, acc_r = parse_rubric_judge_log(pipe_log)
    total_z, correct_z, parseable_z, acc_z, mode_z = parse_zero_shot(
        os.path.join(art, "zero_shot.json")
    )
    rows.append({
        "variant": vid,
        "label": label,
        "rubric_acc": acc_r,
        "rubric_n": total_r,
        "zs_acc": acc_z,
        "zs_total": total_z,
        "zs_parseable": parseable_z,
        "zs_mode": mode_z,
    })

def fmt_pct(x):
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "-"

def fmt_int(x):
    return f"{x}" if isinstance(x, int) else "-"

lines = [
    "# RewardVerse MVP - Visual Quality, 4 Variants Comparison",
    "",
    "Each row is one input-format variant for the same underlying visual-quality",
    "test set (n=30). Rubric Acc is from `judger.py` (auto-rubric); Zero-shot Acc",
    "is the rubric-free baseline run on the same data. Pairwise variants use",
    "two-video rank judging; pointwise variants use single-video 1-5 scoring",
    "(exact-match accuracy).",
    "",
    "| Variant | Description | Rubric Acc | Rubric N | Zero-shot Acc | ZS Parseable / Total | ZS Mode |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    parseable_total = (
        f"{r['zs_parseable']} / {r['zs_total']}"
        if r['zs_parseable'] is not None and r['zs_total'] is not None
        else "-"
    )
    lines.append(
        f"| `{r['variant']}` | {r['label']} | "
        f"{fmt_pct(r['rubric_acc'])} | {fmt_int(r['rubric_n'])} | "
        f"{fmt_pct(r['zs_acc'])} | {parseable_total} | "
        f"{r['zs_mode'] or '-'} |"
    )
lines.append("")
lines.append("Per-variant artifacts:")
for r in rows:
    lines.append(f"- **{r['variant']}** -> `{out_root}/{r['variant']}/`")

pathlib.Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] summary written to {summary_path}")
print("\n".join(lines))
PYEOF

echo
echo "============================================================"
echo " Done. See ${SUMMARY}"
echo "============================================================"
