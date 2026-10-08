# RewardVerse · rm_training — Three-Stage Rubric MLLM Reward Model

> **Status**: 🚧 MVP scaffold (created 2026-06-01). Not yet trained.
>
> Goal: train an in-house **rubric-based MLLM reward model** distilled from
> Gemini, so that the VGRB full eval split (and any other downstream
> benchmark) can be run with **zero API cost** at inference time.

---

## 1. Three stages at a glance

```
┌─────────────── Stage ① Offline Teacher (Gemini, DONE) ─────────┐
│  Status:  Already run offline — produced one rubric scaffold   │
│           per dimension (NOT per pair) as plain text under     │
│           rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt        │
│           (visual_quality uses v_arr_mvp_vq_v2_prompt_pair.txt) │
│  Driver (post-processing):                                     │
│    rm_training/stage1_teacher/build_gold_rubric_dataset_from_txt│
│  Output:  data/rm_train/gold_rubrics_18d.jsonl   (505 rows)    │
│  Each row carries the 5 gold rubric entries as                 │
│           {path, weight=1/K, tips}; weights are equal because  │
│           the v2.txt scaffold has no per-rubric weight field.  │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────── Stage ② Online Training (Student GRPO) ─────────┐
│  Backbone: Qwen2.5-VL-7B-Instruct (+ LoRA)                     │
│  Step (G=8 rollouts per (q, y⁺, y⁻) pair):                     │
│   1) Sample G rubrics R̂_g given (q, caption only).             │
│   2) For each R̂_g, run student in pointwise-score mode TWICE   │
│      — once on y⁺, once on y⁻ — model never sees both videos   │
│      simultaneously (deployment-consistent).                   │
│   3) reward = σ(S⁺ - S⁻) + λ(t)·sim(R̂_g, R*) + β·r_format     │
│      where sim() is bipartite matching (Hungarian) on rubric   │
│      path embeddings; valid_ratio gates rubrics with ≈0 std.   │
│   4) GRPO group advantage over G rollouts → backward.          │
│  Annealing: λ(t) = λ₀ · (1 - valid_ratio_running_mean)         │
│  Cold-start: NO SFT — Instruct base + early-high β + λ anneal. │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────── Stage ③ Deployment (Pointwise RM) ──────────────┐
│  Input :  q + caption + ONE video y                            │
│  Driver:  rm_training/stage3_deploy/pointwise_rm_inference.py  │
│  Logic :  R̂(y)   = student.generate_rubric(q, caption)         │
│           c_k(y) = student.score_video(y, R̂)                   │
│           S(y)   = Σ w_k·c_k / Σ w_k                           │
│  Pairwise eval: score y_a, y_b independently → compare.        │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory layout

```
rm_training/
├── README.md                          ← this file
├── stage0_diagnosis/                  ← gating: is the base model good enough?
│   ├── __init__.py
│   ├── build_pair_data.py             # v_arr_mvp -> exp0 jsonl + URL→local
│   └── exp0_base_diagnosis.py         # D1 pairwise / D2 frame sweep / D3 dim probe
├── stage1_teacher/
│   ├── __init__.py
│   ├── build_gold_rubric_dataset_from_txt.py  # v2.txt scaffolds → jsonl
│   ├── parse_v2_txt.py                         # Theme/Tip text parser
│   └── prompt_templates.py                     # caption synthesis (optional)
├── stage2_grpo/
│   ├── __init__.py
│   ├── train_grpo.py                  # main entrypoint
│   ├── reward_fn.py                   # σ(S⁺-S⁻) + λ·sim + β·fmt
│   ├── student_wrapper.py             # Qwen2.5-VL rubric-gen + score
│   ├── rubric_parser.py               # student JSON → structured
│   ├── gold_align.py                  # bipartite Hungarian sim
│   ├── valid_filter.py                # std≈0 cheat detector
│   ├── data_collator.py
│   └── configs/
│       ├── lora_qwen25vl.yaml
│       └── grpo_default.yaml
├── stage3_deploy/
│   ├── __init__.py
│   ├── pointwise_rm_inference.py      # single-video scorer
│   └── eval_on_vgrb.py                # M1 milestone entrypoint
└── tests/
    ├── test_rubric_parser.py
    ├── test_gold_align.py
    └── test_reward_fn.py

data/rm_train/                          ← Stage ① products live here
├── gold_rubrics_18d.jsonl
└── splits/
    ├── train.jsonl
    └── val.jsonl
```

---

## 3. Concrete dependencies (runtime)

| package | required version | purpose |
|---|---|---|
| `torch` | ≥ 2.2 | already 2.6.0+cu124 |
| `transformers` | ≥ 4.52 | already 4.53.2 (Qwen2.5-VL is supported here) |
| `trl` | ≥ 0.14 | already 0.14.0 (`GRPOTrainer`) |
| `peft` | ≥ 0.13 | already 0.14.0 (LoRA on Qwen2.5-VL) |
| `accelerate` | ≥ 1.0 | already 1.3.0 |
| `vllm` | ≥ 0.7 (optional) | already 0.7.2 (faster score forward in §②) |
| `sentence-transformers` | ≥ 2.7 | **NOT installed** — needed for gold_align.py |
| `scipy` | any | `linear_sum_assignment` (Hungarian) |
| `qwen-vl-utils` | latest | video frame sampling helper |

> Only `sentence-transformers` and `qwen-vl-utils` are missing. Both are
> small CPU-only installs; do NOT block Stage ① on them.

---

## 4. Required external assets (NOT yet present)

| asset | where it should live | how to get it |
|---|---|---|
| **Qwen2.5-VL-7B-Instruct** weights | `/group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct/` | `huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir <above>` (~16 GB) |
| **BGE embedder** for gold_align | `/group/40063/jacuzzitang/models/bge-m3/` (or any sentence-transformer model) | `huggingface-cli download BAAI/bge-m3 --local-dir <above>` (~2.3 GB) |
| **Train videos** for §② | already at `data/v_arr_mvp/videos/...` | reuse existing v_arr_mvp pairwise data |

---

## 5. End-to-end happy-path runbook

### Stage ⓪  (gating, GPU-node, before any training)

Goal: verify Qwen2.5-VL-7B-Instruct can handle long-video pair judgement BEFORE
spending teacher / training budget.

```bash
# 0.1 — prepare exp0 jsonl from existing v_arr_mvp pair files
#       (URL→local mapping reuses runs/baselines/_video_cache; add
#        --prefetch_missing to download whatever is still URL-only).
python rm_training/stage0_diagnosis/build_pair_data.py \
    --pair_glob "data/v_arr_mvp/*_mvp/v2_prompt_pair.json" \
    --out_dir   data/rm_train/exp0 \
    --require_local            # drop pairs that don't resolve locally

# 0.2 — run the three diagnostics (needs GPU + the local Qwen2.5-VL weight)
python rm_training/stage0_diagnosis/exp0_base_diagnosis.py \
    --diagnosis all \
    --model_path /group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \
    --max_samples 200          # smoke; remove for full run
```

Gating verdict from D1's `accuracy_on_decided`:

| Range | Verdict | Action |
|---|---|---|
| `> 0.70`  | **PASS** | proceed to Stage ① + Stage ② |
| `0.65 – 0.70` | **GRAY** | proceed; let rubric reinforce the weak dims surfaced by D3 |
| `< 0.65` | **FAIL** | do NOT enter RL — fix base model / framing / data first |

### Stage ①  (assembly only — Gemini calls already done offline)

Stage ① does NOT call Gemini anymore — the gold rubric scaffolds were
produced once during the v2 offline run and live as plain-text files at
``rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt`` (and
``v_arr_mvp_vq_v2_prompt_pair.txt`` for ``visual_quality``). What remains
is to parse + assemble them into the per-pair training jsonl that
Stage ② consumes:

```bash
python rm_training/stage1_teacher/build_gold_rubric_dataset_from_txt.py \
    --pair_glob   "data/v_arr_mvp/*_mvp/v2_prompt_pair.json" \
    --rubric_dir  rubric_pipeline/rubrics \
    --output      data/rm_train/gold_rubrics_18d.jsonl \
    --split_outdir data/rm_train/splits \
    --val_frac    0.1 \
    --require_local       # drop pairs whose videos can't be resolved locally
```

This emits 505 rows (30 pairs × 18 dims minus a few short dims) in
roughly 1 s on CPU. Run is idempotent — rerun it freely whenever the
v2.txt scaffolds are updated.

### Stage ②  (training, multi-GPU)
```bash
# CPU dummy mode (sanity-check only, no GPU needed)
RM_DUMMY=1 python rm_training/stage2_grpo/train_grpo.py \
    --gold_rubrics data/rm_train/gold_rubrics_18d.jsonl \
    --output_dir   ckpts/student_grpo_dummy/ \
    --num_generations 8 \
    --max_steps 4

# Real run (GPU node, ~4×A100 80G)
accelerate launch rm_training/stage2_grpo/train_grpo.py \
    --student_model /group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \
    --gold_rubrics  data/rm_train/gold_rubrics_18d.jsonl \
    --embed_model   /group/40063/jacuzzitang/models/bge-m3 \
    --output_dir    ckpts/student_grpo/ \
    --lora_config   rm_training/stage2_grpo/configs/lora_qwen25vl.yaml \
    --num_generations 8 \
    --max_steps 1000 --save_every 100 \
    --beta_kl 0.04 --max_grad_norm 1.0 --learning_rate 5e-6 \
    --gradient_checkpointing
```

> **What gets backproped.** The policy gradient flows ONLY through the
> rubric-gen tokens. The two score forwards run greedy + no_grad
> (set `--score_sample` to override) — letting them carry gradient
> would teach the model to fake the score numbers rather than improve
> rubrics. KL is computed against a "reference policy" defined as the
> same model with the LoRA adapter disabled (`disable_adapter()`),
> i.e. the frozen base — no second 7B model is loaded.
>
> **Memory.** With `--gradient_checkpointing` (default ON) plus
> per-rollout backward + grad accumulation, peak activation memory is
> bounded by ONE rollout regardless of `--num_generations`. Use this
> path on single-card runs; disable with `--no-gradient_checkpointing`
> on 4×80GB if you want max throughput.

### Stage ③  (deployment + VGRB eval)
```bash
# Score one video
python rm_training/stage3_deploy/pointwise_rm_inference.py \
    --student_ckpt ckpts/student_grpo/checkpoint-final \
    --query  "A red sports car drifts around a mountain bend" \
    --video  data/v_arr_mvp/videos/varr_seed_001_win.mp4

# VGRB 200-pair smoke evaluation
python rm_training/stage3_deploy/eval_on_vgrb.py \
    --student_ckpt ckpts/student_grpo/checkpoint-final \
    --vgrb_dataset /group/40063/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench/pairwise_view/Overall.json \
    --output runs/external_benchmarks/videogen_rewardbench/student_rm/Overall.json
```

---

## 6. Reward formula (Stage ②)

$$
\text{Reward}(R̂_g) \;=\; \underbrace{\sigma\!\big(S(y^+\!\mid\!R̂_g) - S(y^-\!\mid\!R̂_g)\big)}_{r_\text{pref}}
\;+\; \lambda(t)\cdot \underbrace{\frac{1}{n}\sum_{j}\max_i \cos\!\big(d_i, d^*_j\big)}_{r_\text{align (bipartite)}}
\;+\; \beta\cdot r_\text{format}
$$

- `r_pref` ∈ (0, 1) — soft Bradley-Terry; uses the running score difference,
  not MSE; ties to pairwise GT.
- `r_align` ∈ [0, 1] — bipartite Hungarian cosine over rubric **paths only**,
  routed through a **warm-start anchor**: once `sim ≥ align_floor`
  (default 0.7) the reward saturates at 1.0 and stops contributing
  variance to the GRPO advantage; below the floor it decays linearly to
  0. Treats the gold rubric as an "initial alignment" anchor (don't drift
  too far from teacher's sub-aspect taxonomy) — NOT as a ground-truth
  answer to be copied verbatim. Set `--no-align_saturate` to recover the
  legacy linear behaviour.
- `r_format` ∈ [0, 1] — JSON-validity (rubric-gen) + scaffold-validity
  (score forward) + weight-sum-1 + 5 entries + scores in [1.0, 5.0].
- λ(t) = λ₀ · (1 − valid_ratio_ema); decays from λ₀ ≈ 0.3 to ~0 as the
  student becomes self-sufficient at producing **valid** rubrics
  (defined as `|S⁺ − S⁻| > eps`, eps ≈ 0.05).
- **β(t)** decays from `β_0 ≈ 0.5` to `β_min ≈ 0.05` over
  `--beta_format_decay_steps` (default 500) on a `linear` / `cosine` /
  `step` / `constant` schedule. Early-high β grounds the policy on
  schema compliance (valid rubric + valid scaffold) before the
  pref/align signals dominate; the `β_min` floor keeps a small format
  pressure active so the LM doesn't drift back into bad formats late
  in training.

## Per-dimension prompt scaffolding (Stage ② / ③)

Both rubric-gen and score forwards now inject a per-dim
`DimensionContext` materialised from
`rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml`:

```
You are a strict video reward judge for the **{dimension}** dimension ({Aspect > Main}).

## Task
{task_description (short version, ~700-1500 chars)}

## Sub-aspects (your 5 rubrics MUST collectively cover these)
- Acting-Action-Action-Emotion_Synergy
- Acting-Action-Action_Tension
- ...
```

This keeps Stage ② student rubrics anchored in the same EvalVerse
sub-aspect taxonomy that Stage ① teacher used to generate the gold
rubric scaffolds. Without it, student rubrics drift to generic
"theme: visual quality" boilerplate and `r_align` cannot anchor.

The yaml-derived task descriptions are **shared assets** — Stage ① /
Stage ② / Stage ③ all read the same `task_descriptions.load_task_specs`
function so prompt content can never diverge across stages.

> **Score path (`score_method`).** The score forward emits a plain-text
> scaffold ``theme_1: <X> theme_2: <X> ...``. With `soft_logits`
> (default; matches stage0 D5 ablation winner) we read the per-position
> logits at each `<X>` over the {1..5} digit-id variants and report
> E[k]=Σ k·softmax(logits)_k as a continuous float in [1.0, 5.0]. This
> eliminates parse_fail and the integer-tie problem. With
> `integer_text` we parse the model's text output instead — coarser but
> requires no logits bookkeeping.

---

## 7. Open questions / TODOs

| # | item | who decides |
|---|---|---|
| Q0 | **Has Stage ⓪ run yet?** D1 acc = ? | first physical exam, gates everything else |
| Q1 | Caption source: do we have human captions, or auto-generate via Gemini? | requires user |
| Q2 | Train/val split: pure random vs by-dim stratified? | default: 90/10 by-dim |
| Q3 | LoRA target modules: full attn (q,k,v,o) or +mlp? | start with attn-only |
| Q4 | β decay schedule: linear, cosine, or step? | DONE: `--beta_format_schedule` defaults to linear; cosine/step/constant available |
| Q5 | Whether to re-derive per-pair rubrics later (current scaffold is per-dim, K=5, equal weights from v2.txt) | default: keep per-dim until ablation says otherwise |
| Q6 | When the in-house RM is good enough → run M1 (VGRB full split) | future milestone |

> See [`runs/STATUS_SYNC.md`](../runs/STATUS_SYNC.md) §8.3 M1 for the
> milestone definition.

---

## 8. CPU dummy mode (for fast iteration without GPU)

Set `RM_DUMMY=1` in env. The student is replaced by a **mock model** that:
- generates a fixed JSON rubric template;
- returns deterministic per-criterion scores (e.g. `2.5 + small noise`);
- skips video loading entirely.

This lets you smoke-test the **reward function**, **GRPO advantage
computation**, and **dataloader collation** without paying GPU cost.
The same `train_grpo.py` script switches back to the real student when
`RM_DUMMY` is unset and `--student_model <path>` is provided.
