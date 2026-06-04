#!/usr/bin/env bash
# Run RewardVerse MVP pipeline for all 4 visual-quality input variants and aggregate
# results into a single markdown comparison table.
#
# Variants (built by scripts/v_arr_mvp/build_visual_quality_mvp_dataset.py):
#   v1_prompt_ywin   : Prompt + y+
#   v2_prompt_pair   : Prompt + y+ + y-   (ARR original)
#   v3_ywin_only     : y+ only
#   v4_pair_only     : y+ + y-            (no prompt)
#
# Usage:
#   export OPENAI_API_KEY=<venus_token>            # for /llmproxy chat
#   export VENUS_SECRET_ID=<your_id>               # for video upload preSign
#   export VENUS_SECRET_KEY=<your_key>             #
#   bash scripts/v_arr_mvp/run_all_variants.sh
#
# Optional env overrides:
#   MODEL                       (default: gemini-3.1-pro)
#   BASE_URL                    (default: http://v2.open.venus.oa.com/llmproxy)
#   CONCURRENCY                 (default: 2)
#   DATA_DIR                    (default: data/v_arr_mvp/visual_quality_mvp)
#   OUT_ROOT                    (default: runs/v_arr_mvp)
#   CONFIG_PATH                 (default: rubric_pipeline/config/v_arr_mvp_t2v.yaml)
#   ARR_VIDEO_INPUT_STRATEGY    (default: native_video)
#   ARR_VIDEO_BLOCK_TYPE        (default: gemini_multimodal_url)
#   FORCE_PREFETCH=1            force the legacy frame-prefetch path
#   ONLY                        comma-list to restrict, e.g. ONLY=v1_prompt_ywin,v2_prompt_pair
#
# How videos are sent to venus llmproxy:
#   With the default native_video + gemini_multimodal_url settings, each video
#   (a remote https mp4 in our datasets) is downloaded once to /tmp, uploaded
#   to venus temp storage via preSign + PUT (rubric_pipeline/utils/venus_upload.py),
#   then referenced as a `gemini_multimodal_url` content part. Uploads are
#   cached at runs/v_arr_mvp/_venus_upload_cache.json.
#
#   For the legacy decord path, set:
#     export ARR_VIDEO_INPUT_STRATEGY=uniform_frames
#     export ARR_VIDEO_BLOCK_TYPE=image_url
#     export FORCE_PREFETCH=1   # decord can't read remote URLs

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
SRC_DATA_DIR="${SRC_DATA_DIR:-data/v_arr_mvp/visual_quality_mvp}"
LOCAL_VIDEO_CACHE="${LOCAL_VIDEO_CACHE:-data/v_arr_mvp/cache_videos}"
# DATA_DIR defaults to SRC_DATA_DIR (raw https mp4 URLs); only the legacy
# decord prefetch path rewrites it to a local-mp4 mirror.
DATA_DIR="${DATA_DIR:-${SRC_DATA_DIR}}"
OUT_ROOT="${OUT_ROOT:-runs/v_arr_mvp}"
CONFIG_PATH="${CONFIG_PATH:-rubric_pipeline/config/v_arr_mvp_t2v.yaml}"
PIPELINE="${PIPELINE:-scripts/v_arr_mvp/run_mvp_pipeline.sh}"

# Default = venus native-video upload path.
# Each remote mp4 in the dataset is downloaded to /tmp once, uploaded to
# venus temp storage, then referenced as a gemini_multimodal_url block.
export ARR_VIDEO_INPUT_STRATEGY="${ARR_VIDEO_INPUT_STRATEGY:-native_video}"
export ARR_VIDEO_BLOCK_TYPE="${ARR_VIDEO_BLOCK_TYPE:-gemini_multimodal_url}"
export ARR_VENUS_MODEL="${ARR_VENUS_MODEL:-${MODEL}}"

if [[ "${ARR_VIDEO_BLOCK_TYPE}" == "gemini_multimodal_url" ]]; then
  if [[ -z "${VENUS_SECRET_ID:-}" || -z "${VENUS_SECRET_KEY:-}" ]]; then
    echo "[error] VENUS_SECRET_ID and VENUS_SECRET_KEY are required for the" >&2
    echo "        gemini_multimodal_url upload path. Either export them, or" >&2
    echo "        switch to the legacy decord/frames path with:" >&2
    echo "          export ARR_VIDEO_INPUT_STRATEGY=uniform_frames" >&2
    echo "          export ARR_VIDEO_BLOCK_TYPE=image_url" >&2
    echo "          export FORCE_PREFETCH=1" >&2
    exit 2
  fi
fi

mkdir -p "${OUT_ROOT}" rubric_pipeline/rubrics "${LOCAL_VIDEO_CACHE}" "${DATA_DIR}"

# ---------------- Optional: legacy frame-prefetch path --------------------
# Only triggered when explicitly requested (FORCE_PREFETCH=1) or when the
# user has switched ARR_VIDEO_INPUT_STRATEGY back to uniform_frames.
if [[ "${FORCE_PREFETCH:-0}" == "1" || "${ARR_VIDEO_INPUT_STRATEGY}" == "uniform_frames" ]]; then
  : "${DATA_DIR:=data/v_arr_mvp/visual_quality_mvp_local}"
  mkdir -p "${DATA_DIR}"
  echo "[prefetch] downloading remote mp4 -> ${LOCAL_VIDEO_CACHE}"
  python scripts/v_arr_mvp/prefetch_videos.py \
    --src_dir   "${SRC_DATA_DIR}" \
    --cache_dir "${LOCAL_VIDEO_CACHE}" \
    --out_dir   "${DATA_DIR}"
fi

# variant_id : dataset_filename : human_label
VARIANTS=(
  "v1_prompt_ywin:v1_prompt_ywin.json:Prompt + y+ (Exemplar)"
  "v2_prompt_pair:v2_prompt_pair.json:Prompt + y+ + y- (ARR original)"
  "v3_ywin_only:v3_ywin_only.json:y+ only"
  "v4_pair_only:v4_pair_only.json:y+ + y- (no prompt)"
)

# Optional filter via ONLY=v1_prompt_ywin,v3_ywin_only
ONLY="${ONLY:-}"
should_run() {
  local vid="$1"
  if [[ -z "${ONLY}" ]]; then return 0; fi
  IFS=',' read -ra keep <<< "${ONLY}"
  for k in "${keep[@]}"; do [[ "${k}" == "${vid}" ]] && return 0; done
  return 1
}

# Pre-flight: every requested dataset file must exist.
for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r vid fname _ <<< "${entry}"
  should_run "${vid}" || continue
  if [[ ! -f "${DATA_DIR}/${fname}" ]]; then
    echo "[error] missing dataset: ${DATA_DIR}/${fname}" >&2
    exit 3
  fi
done

# RewardVerse MVP head-to-head: ALL 4 variants are evaluated on the SAME shared
# pointwise test set (single video + prompt -> continuous score in [1, 5]).
# Each variant only differs in the seed dataset used to GENERATE its rubric;
# the rubric is then applied verbatim to this shared test set so PLCC/SRCC/MAE
# are directly comparable across rows.
SHARED_TEST_DATASET="${SHARED_TEST_DATASET:-${DATA_DIR}/shared_test_pointwise.json}"
if [[ ! -f "${SHARED_TEST_DATASET}" ]]; then
  echo "[error] missing shared pointwise test set: ${SHARED_TEST_DATASET}" >&2
  echo "        Re-run scripts/v_arr_mvp/build_visual_quality_mvp_dataset.py to produce it." >&2
  exit 3
fi

echo "============================================================"
echo " RewardVerse MVP : running 4 visual-quality variants"
echo " MODEL        = ${MODEL}"
echo " BASE_URL     = ${BASE_URL}"
echo " CONCURRENCY  = ${CONCURRENCY}"
echo " DATA_DIR     = ${DATA_DIR}"
echo " OUT_ROOT     = ${OUT_ROOT}"
echo " SHARED_TEST  = ${SHARED_TEST_DATASET}"
echo "============================================================"

# ---------------- Step 0: shared pointwise zero-shot baseline ---------------
# Run ONCE on the shared test set; all 4 variants reuse this artefact.
SHARED_BASELINE_JSON="${OUT_ROOT}/shared_zero_shot.json"
if [[ -s "${SHARED_BASELINE_JSON}" ]]; then
  echo "[info] reusing existing shared zero-shot baseline: ${SHARED_BASELINE_JSON}"
else
  echo
  echo "------------------------------------------------------------"
  echo "[shared baseline] pointwise zero-shot on ${SHARED_TEST_DATASET}"
  echo "------------------------------------------------------------"
  python scripts/v_arr_mvp/pointwise_zero_shot_judge.py \
    --test_dataset "${SHARED_TEST_DATASET}" \
    --base_url    "${BASE_URL}" \
    --model       "${MODEL}" \
    --concurrency_limit "${CONCURRENCY}" \
    --output_json "${SHARED_BASELINE_JSON}" \
    2>&1 | tee "${OUT_ROOT}/shared_zero_shot.log"
fi


for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r vid fname label <<< "${entry}"
  should_run "${vid}" || { echo "[skip] ${vid}"; continue; }

  # SEED dataset = variant-specific (used to GENERATE rubric).
  # TEST dataset = shared pointwise (used to APPLY rubric for head-to-head).
  seed_dataset="${DATA_DIR}/${fname}"
  test_dataset="${SHARED_TEST_DATASET}"
  artifact="${OUT_ROOT}/${vid}"
  rubric="rubric_pipeline/rubrics/v_arr_mvp_vq_${vid}.txt"

  mkdir -p "${artifact}"

  echo
  echo "------------------------------------------------------------"
  echo "[run] ${vid}  (${label})"
  echo "      seed_dataset = ${seed_dataset}"
  echo "      test_dataset = ${test_dataset}  (shared)"
  echo "      artifact     = ${artifact}"
  echo "      rubric       = ${rubric}"
  echo "------------------------------------------------------------"

  set +e
  SEED_DATASET="${seed_dataset}" \
  TEST_DATASET="${test_dataset}" \
  RUBRICS_FILE="${rubric}" \
  ARTIFACT_DIR="${artifact}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  BASE_URL="${BASE_URL}" \
  MODEL="${MODEL}" \
  CONCURRENCY="${CONCURRENCY}" \
  SKIP_BASELINE=1 \
  SKIP_POSITIONAL=1 \
  ZERO_SHOT_JSON_OVERRIDE="${SHARED_BASELINE_JSON}" \
  bash "${PIPELINE}" 2>&1 | tee "${artifact}/pipeline.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[warn] variant ${vid} pipeline exited with code ${rc}; continuing." >&2
  fi
done

# ---------------- Aggregate results into one markdown table ----------------
SUMMARY="${OUT_ROOT}/all_variants_summary.md"
python - "$OUT_ROOT" "$SUMMARY" "${SHARED_BASELINE_JSON}" "${VARIANTS[@]}" <<'PYEOF'
import json, os, re, sys, pathlib

out_root = sys.argv[1]
summary_path = sys.argv[2]
shared_baseline_json = sys.argv[3]
entries = sys.argv[4:]

def safe_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def parse_rubric_judge_metrics(art_dir):
    """Read judger.py metrics. Prefer the JSON file; fall back to log."""
    j = safe_load_json(os.path.join(art_dir, "rubric_judge.json"))
    if isinstance(j, dict):
        return j
    log_path = os.path.join(art_dir, "rubric_judge.log")
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    out = {}
    m = re.search(r"Mode:\s*(\S+)", txt)
    if m: out["mode"] = m.group(1)
    m = re.search(r"Total Tests:\s*(\d+)", txt)
    if m: out["total"] = int(m.group(1))
    m = re.search(r"Parseable:\s*(\d+)", txt)
    if m: out["parseable"] = int(m.group(1))
    m = re.search(r"PLCC:\s*([0-9.\-]+)", txt)
    if m:
        try: out["plcc"] = float(m.group(1))
        except ValueError: pass
    m = re.search(r"SRCC:\s*([0-9.\-]+)", txt)
    if m:
        try: out["srcc"] = float(m.group(1))
        except ValueError: pass
    m = re.search(r"MAE:\s*([0-9.\-]+)", txt)
    if m:
        try: out["mae"] = float(m.group(1))
        except ValueError: pass
    return out or None

def fmt_f(x, fmt=".4f"):
    if isinstance(x, (int, float)):
        try:
            return format(float(x), fmt)
        except Exception:
            return "-"
    return "-"

def fmt_int(x):
    return f"{x}" if isinstance(x, int) else "-"

rows = []
for entry in entries:
    vid, fname, label = entry.split(":", 2)
    art = os.path.join(out_root, vid)
    rj = parse_rubric_judge_metrics(art) or {}
    rows.append({
        "variant": vid,
        "label": label,
        "mode": rj.get("mode") or "pointwise",
        "total": rj.get("total"),
        "parseable": rj.get("parseable"),
        "plcc": rj.get("plcc"),
        "srcc": rj.get("srcc"),
        "mae":  rj.get("mae"),
    })

# Shared baseline (single row, applied identically to the same test set).
zs = safe_load_json(shared_baseline_json) or {}
shared_row = {
    "mode": zs.get("mode") or "pointwise",
    "total": zs.get("total"),
    "parseable": zs.get("parseable"),
    "plcc": zs.get("plcc"),
    "srcc": zs.get("srcc"),
    "mae":  zs.get("mae"),
}

lines = [
    "# RewardVerse MVP - Visual Quality, 4 Rubric-Generation Variants vs Shared Baseline",
    "",
    "All 4 rows below evaluate their auto-generated rubric on the SAME shared",
    "pointwise test set `data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json`",
    "(prompt + single video -> continuous score in [1, 5]). Each variant only",
    "differs in the SEED data shown to Gemini during rubric generation. The last",
    "row is the prompt-only zero-shot baseline (NO rubric) on the same test set.",
    "",
    "| Variant | Rubric-generation input | Mode | N | Parseable | PLCC | SRCC | MAE |",
    "|---|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(
        f"| `{r['variant']}` | {r['label']} | {r['mode']} | "
        f"{fmt_int(r['total']) if isinstance(r['total'],int) else '-'} | "
        f"{fmt_int(r['parseable']) if isinstance(r['parseable'],int) else '-'} | "
        f"{fmt_f(r['plcc'])} | {fmt_f(r['srcc'])} | {fmt_f(r['mae'])} |"
    )
lines.append(
    f"| `shared_zero_shot` | (no rubric, prompt-only baseline) | "
    f"{shared_row['mode']} | "
    f"{fmt_int(shared_row['total']) if isinstance(shared_row['total'],int) else '-'} | "
    f"{fmt_int(shared_row['parseable']) if isinstance(shared_row['parseable'],int) else '-'} | "
    f"{fmt_f(shared_row['plcc'])} | {fmt_f(shared_row['srcc'])} | {fmt_f(shared_row['mae'])} |"
)
lines.append("")
lines.append("Per-variant artifacts:")
for r in rows:
    lines.append(f"- **{r['variant']}** -> `{out_root}/{r['variant']}/`")
lines.append(f"- **shared_zero_shot** -> `{shared_baseline_json}`")

pathlib.Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] summary written to {summary_path}")
print("\n".join(lines))
PYEOF

echo
echo "============================================================"
echo " All variants done. See ${SUMMARY}"
echo "============================================================"
