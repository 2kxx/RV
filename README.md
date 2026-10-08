# RewardVerse

<p align="center">
  <a href="https://arxiv.org/abs/2605.08354v1"><img src="https://img.shields.io/badge/arXiv-ARR-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/2kxx/rewardverse"><img src="https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

RewardVerse trains a VLM-based reward model that generates explicit,
query-specific rubrics and scores generated videos / images against them,
then uses that reward to fine-tune a downstream video generation model
via GRPO.

## Repository Layout

```
RewardVerse/
├── rm_training/              # RM training pipeline
│   ├── stage1_teacher/       #  teacher rubric generation & parsing
│   ├── stage2_grpo/          #  student GRPO training (main)
│   ├── stage3_deploy/        #  evaluation on VGRB / external benchmarks
│   └── tests/
├── rubric_pipeline/          # Auto-Rubric generation pipeline (MVP)
├── scripts/
│   ├── train_8gpu_zero3.sh           # Phase A: rubric warm-up (8×GPU, ZeRO-3)
│   ├── train_8gpu_zero3_warmup.sh    # Phase A variant
│   ├── train_8gpu_zero3_joint.sh     # Phase B: joint rubric + score training
│   ├── downstream_grpo/              # downstream GRPO fine-tuning of Wan-2.1
│   ├── eval/                         # EvalVerse / VGRB benchmark evaluation
│   ├── preprocess/                   # embedding preprocessing
│   └── v_arr_mvp/                    # MVP evaluation scripts
├── fastvideo/                # vendored FastVideo components
├── evalverse_evaluator_integration/  # EvalVerse evaluator bridge
├── judger.py                 # standalone pointwise inference entrypoint
├── data/
│   ├── rm_train/gold_rubrics_18d.jsonl   # 18-dim gold rubric training data
│   └── v_arr_mvp/            # MVP test datasets (per-dimension)
├── docs/
└── examples/
```

## Environment

Validated on Python 3.10, CUDA 12.4, torch 2.6.0, transformers 4.51.3,
deepspeed 0.16.5, flash-attn 2.7.4.post1. All versions pinned in
`requirements.txt` / `pyproject.toml`.

```bash
cd RewardVerse
bash install_rm_env.sh --conda rewardverse
conda activate rewardverse
```

Flash-attn is compiled by default (first time ~5-10 min, needs nvcc on PATH).
Use `--no-flash` to skip; use `--extras vllm` to also install vLLM for the
rubric generation pipeline.

## RM Training

### Phase A — Rubric warm-up

Trains the score head against gold rubrics only (no rubric forward):

```bash
bash scripts/train_8gpu_zero3.sh
# or the variant:
bash scripts/train_8gpu_zero3_warmup.sh
```

### Phase B — Joint rubric + score GRPO

Resumes from a Phase-A checkpoint and trains both legs jointly
(rubric forward + score PG on student rubric):

```bash
bash scripts/train_8gpu_zero3_joint.sh \
    ckpts/student_grpo_8gpu_zero3_warmup/checkpoint-60 \
    ckpts/student_grpo_8gpu_zero3_joint_v3
```

The final released checkpoint is available on
[HuggingFace](https://huggingface.co/2kxx/rewardverse).

### Configuration

Deepspeed configs live in `rm_training/stage2_grpo/configs/`.
Key hyperparameters (LoRA rank, KL coefficients, score_pg schedule)
are set as CLI arguments in the training scripts and documented in
`docs/rgpo_method.md`.

## Downstream GRPO

Fine-tunes Wan-2.1-1.3B with RewardVerse or baselines as reward:

```bash
cd scripts/downstream_grpo
cat README.md   # detailed 4-arm experiment guide
```

Four arms: unconstrained VideoScore, rubric zero-shot (no training),
RewardVerse (trained RGPO), and VideoAlign (appendix). The reward model
runs on a separate GPU and communicates via Unix socket.

## Evaluation

```bash
# Standalone pointwise scoring
python judger.py --video_path <video> --prompt <prompt> \
    --ckpt 2kxx/rewardverse

# VGRB benchmark
bash scripts/eval/run_vgrb_overall_dp.sh

# EvalVerse multidim evaluation
python scripts/eval/evalverse_multidim_eval.py --help

# Full downstream eval (VBench + Gemini judge + figures)
bash scripts/downstream_grpo/eval/run_all_eval.sh
```

## Data

- `data/rm_train/gold_rubrics_18d.jsonl` — gold rubric training set
  (18 dimensions, ~2.7 MB). Used by Phase A/B training and the
  downstream GRPO prompt pool.
- `data/v_arr_mvp/` — MVP test datasets, one sub-directory per
  dimension (`action`, `character`, `chromaticity`, …), each containing
  `test_pointwise_*.json` and a `manifest.json`. Video files are not
  included in the repo; place them under `data/v_arr_mvp/videos/` or
  update the paths in the manifests.

## Citation

```bibtex
@misc{tian2026autorubricrewardimplicitpreferences,
      title={Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria},
      author={Juanxi Tian and Fengyuan Liu and Jiaming Han and Yilei Jiang and Yongliang Wu and Yesheng Liu and Haodong Li and Furong Xu and Wanhua Li},
      year={2026},
      eprint={2605.08354},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.08354},
}
```
