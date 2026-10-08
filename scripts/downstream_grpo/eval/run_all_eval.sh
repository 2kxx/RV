#!/usr/bin/env bash
# End-to-end evaluation for all four arms:
#   1. generate 200 videos per arm on the held-out eval prompts
#   2. run VBench (or fallback) on each arm's videos
#   3. run Gemini pairwise judge (arm vs Base)
#   4. aggregate training jsonls into the reward-hacking figure
#   5. build the final §5.4 LaTeX table
#
# Run from repo root. Requires:
#   - trained ckpts under ckpts/downstream/arm_*/checkpoint-final
#   - OPENAI_API_KEY (Gemini) exported

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CFG=scripts/downstream_grpo/configs/wan_grpo_default.yaml
SPLITS=scripts/downstream_grpo/configs/prompts/evalverse_eval.jsonl
RUNS_ROOT=runs/downstream

ARMS=("arm_A_base" "arm_B_videoscore" "arm_C_v3zeroshot" "arm_D_rewardverse")
CKPTS=("" "ckpts/downstream/arm_B_videoscore/checkpoint-final" \
       "ckpts/downstream/arm_C_v3zeroshot/checkpoint-final" \
       "ckpts/downstream/arm_D_rewardverse/checkpoint-final")

for i in "${!ARMS[@]}"; do
    arm=${ARMS[$i]}
    ckpt=${CKPTS[$i]}
    outdir="${RUNS_ROOT}/eval_videos/${arm}"
    if [ -d "$outdir" ] && [ "$(ls -A "$outdir" 2>/dev/null)" ]; then
        echo "[eval_all] videos exist for ${arm}, skipping generate"
    else
        echo "[eval_all] generating videos for ${arm}"
        python -m scripts.downstream_grpo.eval.generate_eval_videos \
            --config "$CFG" \
            --ckpt "$ckpt" \
            --splits "$SPLITS" \
            --out "$outdir"
    fi
done

for arm in "${ARMS[@]}"; do
    echo "[eval_all] VBench for ${arm}"
    python -m scripts.downstream_grpo.eval.run_vbench \
        --videos "${RUNS_ROOT}/eval_videos/${arm}" \
        --splits "$SPLITS" \
        --out    "${RUNS_ROOT}/${arm}_vbench.json"
done

for arm in "${ARMS[@]}"; do
    if [ "$arm" = "arm_A_base" ]; then continue; fi
    echo "[eval_all] Gemini judge for ${arm} vs Base"
    python -m scripts.downstream_grpo.eval.run_gemini_judge \
        --arm_videos  "${RUNS_ROOT}/eval_videos/${arm}" \
        --base_videos "${RUNS_ROOT}/eval_videos/arm_A_base" \
        --splits "$SPLITS" \
        --out    "${RUNS_ROOT}/${arm}_gemini.json"
done

echo "[eval_all] plotting reward-hacking curves"
python -m scripts.downstream_grpo.eval.plot_hacking_curves \
    --log_dir logs/downstream \
    --out_pdf "${RUNS_ROOT}/fig_downstream_hacking.pdf" \
    --out_json "${RUNS_ROOT}/hacking_delta.json"

echo "[eval_all] building LaTeX table"
python -m scripts.downstream_grpo.eval.build_downstream_table \
    --runs_root "${RUNS_ROOT}" \
    --out       "${RUNS_ROOT}/tab_downstream.tex"

echo "[eval_all] done. Copy artifacts:"
echo "  cp ${RUNS_ROOT}/fig_downstream_hacking.pdf RewardVerse/figs/"
echo "  # then paste ${RUNS_ROOT}/tab_downstream.tex into iclr2026_conference.tex"
