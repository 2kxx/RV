# ARR ↔ EvalVerseEvaluator integration

Drop-in glue for plugging the ARR / RewardVerse MVP Auto-Rubric judge into the
EvalVerseEvaluator framework. The result is a per-dimension EvalVerseEvaluator
that uses ARR's pre-baked rubric text + frozen VLM judge to produce a
score in `[0, 5]` per video, with `task_name`-aware routing for the t2v / r2v
family.

```
evalverse_evaluator_integration/
├── README.md
├── video_eval/
│   └── evaluators/
│       └── arr_evaluator.py        ← copy to EvalVerseEvaluator/video_eval/evaluators/
└── configs/
    └── eval_with_arr.yaml          ← merge into EvalVerseEvaluator/configs/eval.yaml
```

## Why this design

- EvalVerseEvaluator drives the benchmark loop (data loading, model sharing,
  reporting) and expects pointwise `[0, 5]` scores per dimension.
- ARR (RewardVerse MVP / EvalVerse rubric) is the *scoring engine*: rubric +
  Gemini-3 / Qwen3-VL judge in pointwise mode.
- Each EvalVerseEvaluator dimension is bound to one rubric text file.
  ARR generates these files **offline**, once, from labeled seed data.
- `task_name` from `VideoData.metadata` routes the sample into the correct ARR
  task family: T2V keeps only prompt+video; R2V (subject / scene / object)
  attaches the reference image; camera_r2v attaches `ref_video_path` as the
  ARR ground-truth video channel.

## Files

| Path | Role |
| --- | --- |
| `video_eval/evaluators/arr_evaluator.py` | `ARRPointwiseEvaluator(BaseEvaluator)`, registered with `@EVALUATORS.register_module()`. |
| `configs/eval_with_arr.yaml` | Sample dimension entries; copy what you need into your real `configs/eval.yaml`. |

In the ARR repo (this branch), the offline rubric generator lives at
`scripts/v_arr_mvp/generate_per_dim_rubrics.py`. See
[`docs/evalverse_evaluator_integration.md`](../docs/evalverse_evaluator_integration.md)
for the full integration recipe.

## Quick start

```bash
# 1. From the ARR repo root, copy the integration files into your
#    EvalVerseEvaluator checkout.
cp evalverse_evaluator_integration/video_eval/evaluators/arr_evaluator.py \
   /path/to/EvalVerseEvaluator/video_eval/evaluators/

# 2. In EvalVerseEvaluator/video_eval/evaluators/__init__.py, add:
#    from .arr_evaluator import ARRPointwiseEvaluator

# 3. Merge `configs/eval_with_arr.yaml` into EvalVerseEvaluator/configs/eval.yaml.

# 4. Generate per-dimension rubric files once (from the ARR repo root):
export OPENAI_API_KEY=<your_api_key>
python scripts/v_arr_mvp/generate_per_dim_rubrics.py \
  --seed_dataset examples/evalverse_raw_pointwise.json \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml \
  --output_dir rubric_pipeline/rubrics

# 5. Point EvalVerseEvaluator at the ARR repo and rubric files via env vars:
export ARR_REPO_ROOT=/abs/path/to/ARR
export ARR_CONFIG_PATH=/abs/path/to/ARR/rubric_pipeline/config/v_arr_mvp_t2v.yaml
export ARR_RUBRICS_DIR=/abs/path/to/ARR/rubric_pipeline/rubrics

# 6. Run EvalVerseEvaluator as usual.
bash /path/to/EvalVerseEvaluator/scripts/run_eval.sh
```

## Configuration knobs

`ARRPointwiseEvaluator` accepts the following kwargs (and the matching env vars
are used as fallbacks):

| Constructor kwarg | Env var | Effect |
| --- | --- | --- |
| `arr_config_path` | `ARR_CONFIG_PATH` | YAML for the ARR Judger (model, base_url, eval_mode, ...). |
| `arr_rubrics_file` | `ARR_RUBRICS_FILE` | Specific rubric text file. |
| `arr_rubrics_dir` | `ARR_RUBRICS_DIR` | Directory; rubric file resolved as `<dim>.txt`. |
| `arr_repo_root` | `ARR_REPO_ROOT` | Prepended to `sys.path` for the `judger` import. |
| `base_url` | `ARR_BASE_URL` | Override the OpenAI-compat endpoint at runtime. |
| `score_range` | `ARR_SCORE_RANGE` | `"1-5"` (default, ARR raw clamped) or `"0-5"` (linear map 1→0, 5→5). |
| `skip_without_rubrics` | n/a | Return `score=None` when no rubric file is resolved (default `True`). |

## Notes & caveats

- **Model sharing**: the framework's `model` instance (e.g. shared `GeminiApi`)
  is currently ignored. ARR constructs its own OpenAI-compat client from
  `arr_config_path`. Point both to the same upstream endpoint to share quota
  rather than CPU memory.
- **Async**: `Judger.initialize()` and `Judger._evaluate_item()` are async. The
  adapter calls them through `_run_coro`, which transparently launches a
  dedicated event loop when invoked inside an existing one.
- **One rubric per dimension**: ARR scoring is rubric-bound. Each
  EvalVerseEvaluator dimension needs its own pre-generated rubric text file.
- **R2V family**: `task_name` controls reference routing. `subject_r2v` /
  `scene_r2v` / `object_r2v` use the first entry of `ref_image_paths`;
  `camera_r2v` uses `ref_video_path`. Other task names fall back to T2V.
