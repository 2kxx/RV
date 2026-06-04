#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-click builder for the Visual-Quality MVP dataset (4 input variants).
#
# Pipeline:
#   1) Mine 30 strict VQ-dominance pairs from all `merged_evaluate_results_*.jsonl`
#      parts (winner strictly beats loser on Physics / Rendering / Temporal).
#   2) Apply the offline manual EN translations stored in
#      `data/v_arr_mvp/visual_quality_mvp/_prompt_translations.json`.
#      No external API is called.
#   3) Emit 4 variant JSONs that match the test-time format:
#        v1_prompt_ywin.json   (Prompt + y+,  pointwise)
#        v2_prompt_pair.json   (Prompt + y+ + y-, pairwise)
#        v3_ywin_only.json     (y+ only, pointwise, no prompt)
#        v4_pair_only.json     (y+ + y-, pairwise, no prompt)
#   4) Sanity-check every sample for strict 3-sub-dim dominance and report
#      score-gap / method-pair statistics.
#
# Usage:
#   bash scripts/v_arr_mvp/oneclick_build_vq_mvp.sh
#   # or pass overrides:
#   N_PAIRS=50 OUT_DIR=data/v_arr_mvp/vq_mvp_v2 \
#       bash scripts/v_arr_mvp/oneclick_build_vq_mvp.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve paths relative to repo root (parent of this script's dir / dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# -------- knobs (override via env) ---------------------------------------
N_PAIRS="${N_PAIRS:-30}"
OUT_DIR="${OUT_DIR:-data/v_arr_mvp/visual_quality_mvp}"
INPUT_GLOB="${INPUT_GLOB:-merged_evaluate_results_with_path_part*.jsonl}"
TRANSLATION_CACHE="${TRANSLATION_CACHE:-${OUT_DIR}/_prompt_translations.json}"
DRY_DIR="${DRY_DIR:-${OUT_DIR}_raw_zh}"
PYTHON="${PYTHON:-python}"

echo "=========================================================="
echo " RewardVerse Visual-Quality MVP : one-click builder"
echo "=========================================================="
echo " repo root         : ${REPO_ROOT}"
echo " input glob        : ${INPUT_GLOB}"
echo " number of pairs   : ${N_PAIRS}"
echo " output dir        : ${OUT_DIR}"
echo " dry-run dir (zh)  : ${DRY_DIR}"
echo " translation cache : ${TRANSLATION_CACHE}"
echo "=========================================================="

mkdir -p "${OUT_DIR}"

# ----------------------------------------------------------------------------
# Step 1 : strict mining + (placeholder) raw-zh JSON dump
# ----------------------------------------------------------------------------
echo
echo "[1/3] mining ${N_PAIRS} strict VQ-dominance pairs (no translation)…"
"${PYTHON}" scripts/v_arr_mvp/build_visual_quality_mvp_dataset.py \
    --input  "${INPUT_GLOB}" \
    --max_pairs "${N_PAIRS}" \
    --strict_only \
    --no_translate \
    --output_dir "${DRY_DIR}"

# ----------------------------------------------------------------------------
# Step 2 : apply offline translation cache + write final 4 variants
# ----------------------------------------------------------------------------
echo
echo "[2/3] applying offline EN translation cache…"
"${PYTHON}" - <<PY
import json, sys
from pathlib import Path

src_dir   = Path("${DRY_DIR}")
dst_dir   = Path("${OUT_DIR}")
cache_p   = Path("${TRANSLATION_CACHE}")

dst_dir.mkdir(parents=True, exist_ok=True)

if not cache_p.exists():
    sys.exit(f"[error] translation cache not found: {cache_p}\n"
             f"        please populate it (zh -> en) before running this step.")

cache = json.loads(cache_p.read_text(encoding="utf-8"))
print(f"  cache entries                   : {len(cache)}")

v2 = json.loads((src_dir / "v2_prompt_pair.json").read_text(encoding="utf-8"))
miss = [it["metadata"]["prompt_zh"]
        for it in v2
        if it["metadata"]["prompt_zh"] not in cache]
print(f"  zh prompts needing translation   : {len(v2)}")
print(f"  cache misses                     : {len(miss)}")
if miss:
    print("  first 3 missing prompts (truncated):")
    for m in miss[:3]:
        print(f"    -- {m[:80]}")
    sys.exit("[error] cache does not cover every mined pair; "
             "extend _prompt_translations.json and re-run.")

def patch(items, *, prompted: bool):
    out = []
    for it in items:
        new = dict(it)
        if prompted:
            new["query"] = cache[it["metadata"]["prompt_zh"]]
        # no-prompt variants already carry the placeholder query.
        out.append(new)
    return out

variants = [
    ("v1_prompt_ywin.json", True),
    ("v2_prompt_pair.json", True),
    ("v3_ywin_only.json",   False),
    ("v4_pair_only.json",   False),
]
for name, prompted in variants:
    items = json.loads((src_dir / name).read_text(encoding="utf-8"))
    items = patch(items, prompted=prompted)
    target = dst_dir / name
    target.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"  wrote {len(items):>2d} -> {target}")

# Refresh manifest.
mani = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
mani["translate"]            = True
mani["translation_source"]   = "manual offline cache"
mani["translation_cache"]    = str(cache_p)
mani["translation_model"]    = None
(dst_dir / "manifest.json").write_text(
    json.dumps(mani, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"  wrote manifest -> {dst_dir / 'manifest.json'}")
PY

# ----------------------------------------------------------------------------
# Step 3 : sanity check + summary
# ----------------------------------------------------------------------------
echo
echo "[3/3] sanity-checking final dataset…"
"${PYTHON}" - <<PY
import json, re
from collections import Counter
from pathlib import Path

dst_dir = Path("${OUT_DIR}")
SUBS = ["Aesthetics-Visual_Quality-Physics",
        "Aesthetics-Visual_Quality-Rendering_Quality",
        "Aesthetics-Visual_Quality-Temporal_Consistency"]

ok = True
for name in ["v1_prompt_ywin.json", "v2_prompt_pair.json",
             "v3_ywin_only.json",   "v4_pair_only.json"]:
    p = dst_dir / name
    if not p.exists():
        print(f"  [missing] {p}")
        ok = False
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    sample = data[0]
    print(f"  {name:<24s} samples={len(data):<3d} keys={list(sample.keys())}")

# Strict dominance + stats on v2.
data = json.loads((dst_dir / "v2_prompt_pair.json").read_text(encoding="utf-8"))
violations = []
gaps = []
task_dist = Counter()
mp_dist = Counter()
for s in data:
    w, l = s["metadata"]["winner_sub_scores"], s["metadata"]["loser_sub_scores"]
    if not all(w[k] > l[k] for k in SUBS):
        violations.append(s["id"])
    gaps.append(s["metadata"]["score_gap"])
    task_dist[s["task_type"]] += 1
    mp_dist[(s["metadata"]["winner_method"], s["metadata"]["loser_method"])] += 1

print()
print(f"  total samples                    : {len(data)}")
print(f"  strict 3-sub-dim dominance       : {'PASS' if not violations else 'FAIL ' + str(violations)}")
print(f"  score-gap min / mean / max       : "
      f"{min(gaps):.3f} / {sum(gaps)/len(gaps):.3f} / {max(gaps):.3f}")
print(f"  task_type distribution           : {dict(task_dist)}")
print(f"  top-5 winner > loser method pairs:")
for (wm, lm), n in mp_dist.most_common(5):
    print(f"    {wm:<25s} > {lm:<25s} : {n}")
PY

echo
echo "=========================================================="
echo " Done. Final dataset @ ${OUT_DIR}"
echo "=========================================================="
ls -la "${OUT_DIR}"
