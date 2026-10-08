# Downstream GRPO Experiment — §5.4 of the paper

This directory contains everything needed to reproduce the four-arm GRPO
fine-tuning of `Wan-2.1-1.3B` reported in §5.4 of the RewardVerse paper,
and to build both the main table and the reward-hacking diagnostic
figure directly from the training logs.

Everything under `src/` and `launch/` is designed to run on a single
8×H100 node. `eval/` is CPU/GPU-light and can be run on a smaller box
after rsyncing the generated eval videos back.

## Directory map

```
scripts/downstream_grpo/
├── configs/
│   ├── ds_zero2.json                      # DeepSpeed ZeRO-2 config
│   ├── wan_grpo_default.yaml              # GRPO hyperparams
│   └── prompts/                            # populated on first launch:
│       ├── evalverse_train.jsonl          # 640 stratified prompts
│       ├── evalverse_eval.jsonl           # 200 held-out
│       └── evalverse_fixed32.jsonl        # 32-prompt fixed eval batch
├── src/
│   ├── prompt_pool.py                     # loads gold_rubrics_18d.jsonl → splits
│   ├── sampler_wan.py                     # Wan-2.1 rollout + per-step logπ
│   ├── train_grpo_wan.py                  # main trainer (DeepSpeed ZeRO-2 + LoRA)
│   ├── reward_adapters.py                 # unified reward adapter interface
│   ├── reward_server.py                   # dedicated-GPU reward server
│   ├── reward_client.py                   # trainer-side client
│   ├── logging_utils.py                   # jsonl + optional wandb logger
│   └── reward_models/
│       ├── rewardverse_adapter.py         # Judger + trained RGPO LoRA
│       ├── v3_zeroshot_adapter.py         # Judger + base Qwen2.5-VL + seed rubric
│       ├── videoscore_v2_adapter.py       # TIGER-Lab/VideoScore-v1.1
│       └── videoalign_adapter.py          # KwaiVGI/VideoReward (optional Arm E)
├── launch/
│   ├── _run_arm.sh                        # shared launcher (called by run_arm_*)
│   ├── run_arm_B_videoscore.sh            # unconstrained pointwise RM
│   ├── run_arm_C_v3zeroshot.sh            # rubric + soft-logits, no training
│   ├── run_arm_D_rewardverse.sh           # main arm — trained RGPO
│   ├── run_arm_E_videoalign.sh            # optional appendix arm
│   └── run_all.sh                         # runs D → B → C sequentially
├── eval/
│   ├── generate_eval_videos.py            # 200 videos from a ckpt on the eval split
│   ├── run_vbench.py                      # VBench-Long + VBench-Text
│   ├── run_gemini_judge.py                # order-swap pairwise win-rate vs Base
│   ├── plot_hacking_curves.py             # fig_downstream_hacking.pdf
│   ├── build_downstream_table.py          # tab_downstream.tex
│   └── run_all_eval.sh                    # end-to-end eval orchestrator
├── env_setup.sh
└── README.md
```

## Quick start (8×H100 node)

```bash
# 0) One-time env setup (idempotent)
cd /path/to/RewardVerse
conda activate rewardverse
bash scripts/downstream_grpo/env_setup.sh

# 1) Point the RewardVerse adapter at your trained RGPO checkpoint.
#    If you don't have one yet, this can stay unset — the adapter will
#    transparently fall back to V3-zeroshot so the pipeline still runs.
export RM_CKPT=/path/to/ckpts/student_grpo_.../checkpoint-final

# 2) Smoke test (30 steps on Arm D only, ~30 min)
MAX_STEPS=30 EVAL_EVERY=10 SAVE_EVERY=30 \
    bash scripts/downstream_grpo/launch/run_arm_D_rewardverse.sh

# 3) Full run: 3 arms × 500 steps ≈ 30 GPU-hours
bash scripts/downstream_grpo/launch/run_all.sh

# 4) Evaluate all arms (VBench + Gemini + hacking curves + table)
export OPENAI_API_KEY=<your gemini key>
bash scripts/downstream_grpo/eval/run_all_eval.sh

# 5) Paper build
cp runs/downstream/fig_downstream_hacking.pdf RewardVerse/figs/
# then paste runs/downstream/tab_downstream.tex into iclr2026_conference.tex
```

## Prompt pool

The trainer draws prompts directly from `data/rm_train/gold_rubrics_18d.jsonl`,
which is already in English and pre-tagged by EvalVerse dimension.
On the first launch the pool is deterministically split into
`640 / 200 / 32` stratified across the 16 dims we care about
(`configs/wan_grpo_default.yaml::dims_whitelist`).

If the 32-prompt eval batch feels too small you can raise
`fixed_eval_size` in the yaml; the splits regenerate whenever the
target jsonls are missing.

## Reward server model

The reward model runs in a **separate process** on GPU 7 and speaks to
the trainer over a Unix socket. This is not just for tidiness —
Qwen2.5-VL-7B + LoRA weighs ~16 GB by itself, and letting it share
the FSDP allocator with the diffusion transformer caused NCCL init
races in the RM training runs (see the notes in
`rm_training/stage2_grpo/eval_ckpt.py`).

Swap the reward model by changing a single argument
(`RM_TYPE=videoscore_v2`, `RM_TYPE=v3_zeroshot`, etc.). All adapters
implement the same `score_batch(list[RewardRequest]) → Tensor[N]`
interface.

## Failure modes

| symptom | likely cause | fix |
|---|---|---|
| Socket doesn't come up | reward server crashed loading its RM | `tail logs/downstream/<arm>_rewardserver.log` |
| Trainer OOM on GPU 0 | `num_generations=8` too large for your VRAM | drop to `G=4` or turn `lora_rank` down |
| VBench crashes on install | `vbench` optional dep failed | the runner auto-falls back to a CLIP+consistency proxy; document this in the paper if used |
| Gemini judge returns no verdicts | rate-limited or missing key | export `OPENAI_API_KEY`; drop concurrency to 2 |
| Nondeterministic prompt splits | `configs/prompts/*.jsonl` didn't exist and re-generated with a new seed | delete them and rerun; the seed comes from `wan_grpo_default.yaml::seed` |

## What gets logged

Every training step writes a line to `logs/downstream/<arm>.jsonl`:
`{"event":"step", "step":N, "mean_reward":..., "reward_std":..., "loss":..., "kl":..., "ratio":..., "grad_norm":..., "seconds":...}`.
Every `eval_every` steps adds `{"event":"eval_fixed", ...}`.

`plot_hacking_curves.py` reads these directly — no `wandb` required
for the figure to build. `wandb` is used only for live monitoring if
`WANDB_PROJECT` is exported.
