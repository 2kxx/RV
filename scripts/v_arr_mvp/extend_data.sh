#!/usr/bin/env bash
# Extend the RewardVerse v_arr_mvp dataset to support the 2026-06-18 plan:
#
#   * Re-mine v2_prompt_pair.json for the four LOW-COVERAGE training dims
#     (logic, rhythm, action, expression) with --max_pairs_per_prompt=2 and
#     a 0.05 score-gap floor, so each dim has 30 pairs ranked by gap.
#     OTHER 12 DIMS ARE NOT TOUCHED — their existing pairs remain frozen.
#
#   * Re-mine test_pointwise_<dim>.json for ALL 16 vision dims at
#     target_count=63, giving ~1008 evaluation samples that keep the
#     5-band stratified distribution. The two audio dims (soundscape,
#     vocal) are skipped because Qwen2.5-VL has no audio path.
#
#   * Rebuild the gold-rubric jsonl from those pair files and write
#     gold_rubrics_18d.clean.jsonl by passing --require_local so any pair
#     whose video isn't in the local cache is dropped automatically.
#
#   * Prefetch every remote .mp4 referenced by training pairs + test set
#     into data/v_arr_mvp/cache_videos/ so resolve_local() succeeds end
#     to end. Concurrency defaults to 10 parallel HTTP downloads.
#
# Inputs (verified 2026-06-18):
#   /workspace/user_code/40063/shuaiili/scripts/RL/evalverse/labeled_data/merged
#       merged_evaluate_results_with_path_part{1..5}.jsonl  (130 records)
#
# Outputs:
#   data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json            (4 dims rewritten)
#   data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json      (16 dims rewritten)
#   data/rm_train/gold_rubrics_18d.jsonl                    (rebuilt)
#   data/rm_train/gold_rubrics_18d.clean.jsonl              (rebuilt, --require_local)
#   data/v_arr_mvp/cache_videos/                            (~2k videos)
#
# Usage:
#   bash scripts/v_arr_mvp/extend_data.sh
#   # OR with custom merged-dir:
#   MERGED_DIR=/some/other/path bash scripts/v_arr_mvp/extend_data.sh
#
# This script is RE-RUNNABLE. Pre-existing cached videos are skipped on
# the prefetch step. v2_prompt_pair.json regeneration is deterministic
# (seed=0 in build_dim_seed_v2.py).

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Egress proxy.
# This host can't reach *.myqcloud.com directly — egress goes through
# the Tencent corporate proxy. We set both lower and uppercase forms
# (urllib reads UPPERCASE; some libs read lowercase). NO_PROXY excludes
# internal mirrors so they don't double-tunnel.
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="${no_proxy:-.woa.com,localhost,127.0.0.1,.oa.com,mirrors.tencent.com,mirrors.cloud.tencent.com}"
export NO_PROXY="$no_proxy"

MERGED_DIR="${MERGED_DIR:-/workspace/user_code/40063/shuaiili/scripts/RL/evalverse/labeled_data/merged}"
MERGED_GLOB="$MERGED_DIR/merged_evaluate_results_with_path_part*.jsonl"
LOW_COVERAGE_DIMS=(logic rhythm action expression)
TEST_DIMS=(action character chromaticity composition consistency expression \
           grounding lens lighting logic materiality pacing progression \
           rhythm scene visual_quality)
TEST_TARGET_COUNT=63
PREFETCH_CONCURRENCY="${PREFETCH_CONCURRENCY:-10}"

if ! ls $MERGED_GLOB >/dev/null 2>&1; then
  echo "[extend] ERROR: source jsonl not found under $MERGED_DIR" >&2
  exit 2
fi

echo "[extend] step 1/4 — re-mining 4 low-coverage training dims (K=2/prompt, gap≥0.05)"
for dim in "${LOW_COVERAGE_DIMS[@]}"; do
  echo "[extend]   dim=$dim"
  python3 scripts/v_arr_mvp/build_dim_seed_v2.py \
    --dimension "$dim" \
    --input "$MERGED_GLOB" \
    --output_dir "data/v_arr_mvp/${dim}_mvp" \
    --max_pairs 30 \
    --max_pairs_per_prompt 2 \
    --min_score_gap 0.05 \
    --no_translate \
    --seed 0
done

echo "[extend] step 2/4 — re-mining 16-dim test_pointwise (target=$TEST_TARGET_COUNT each)"
for dim in "${TEST_DIMS[@]}"; do
  echo "[extend]   dim=$dim"
  python3 scripts/v_arr_mvp/build_dim_seed_v2.py \
    --dimension "$dim" \
    --mode test_pointwise \
    --input "$MERGED_GLOB" \
    --output_dir "data/v_arr_mvp/${dim}_mvp" \
    --target_count "$TEST_TARGET_COUNT" \
    --max_per_record 5 \
    --no_translate \
    --seed 0
done

echo "[extend] step 3/4 — prefetching all remote .mp4 ($PREFETCH_CONCURRENCY workers)"
python3 scripts/v_arr_mvp/prefetch_all_videos.py \
  --cache_dir data/v_arr_mvp/cache_videos \
  --concurrency "$PREFETCH_CONCURRENCY" \
  --manifest data/v_arr_mvp/cache_videos/_prefetch_manifest.json

echo "[extend] step 4/4 — rebuilding gold_rubrics_18d.jsonl + .clean.jsonl"
# We DO NOT carve a held-out val split. GRPO trains on the full 480 rows
# and validation is done against the per-dim test_pointwise_<dim>.json
# pool (~1k rows total) via run_pointwise_eval. Setting --val_frac 0
# disables the splits/{train,val}.jsonl outputs.
# Raw (everything, no local check) — kept for diff/debug.
python3 rm_training/stage1_teacher/build_gold_rubric_dataset_from_txt.py \
  --pair_glob "data/v_arr_mvp/*_mvp/v2_prompt_pair.json" \
  --output    data/rm_train/gold_rubrics_18d.jsonl \
  --val_frac  0 \
  --seed      0 \
  --exclude_dims soundscape vocal

# Cleaned: drops any row whose y_pos / y_neg can't be resolved locally.
# This replaces the old hand-edited .clean.jsonl with a deterministic build.
python3 rm_training/stage1_teacher/build_gold_rubric_dataset_from_txt.py \
  --pair_glob "data/v_arr_mvp/*_mvp/v2_prompt_pair.json" \
  --output    data/rm_train/gold_rubrics_18d.clean.jsonl \
  --val_frac  0 \
  --seed      0 \
  --require_local \
  --exclude_dims soundscape vocal

echo "[extend] DONE."
echo "[extend] sanity: per-dim counts in gold_rubrics_18d.clean.jsonl"
python3 - <<'PY'
import json, collections
c = collections.Counter()
with open('data/rm_train/gold_rubrics_18d.clean.jsonl') as f:
    for line in f:
        c[json.loads(line)['dimension']] += 1
print(f"  total = {sum(c.values())}, dims = {len(c)}")
for d, n in sorted(c.items()):
    print(f"    {d:<22s} {n}")
PY
echo "[extend] sanity: test_pointwise totals"
python3 - <<'PY'
import json, os, glob
total = 0
for fp in sorted(glob.glob('data/v_arr_mvp/*_mvp/test_pointwise_*.json')):
    if fp.endswith('.manifest.json'): continue
    n = len(json.load(open(fp)))
    total += n
    name = fp.split('/')[-2].replace('_mvp','')
    print(f"  {name:<22s} {n:4d}  ({fp})")
print(f"  TOTAL = {total}")
PY
