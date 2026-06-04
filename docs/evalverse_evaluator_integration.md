# EvalVerseEvaluator integration

Run the ARR / RewardVerse Auto-Rubric judge inside the EvalVerseEvaluator benchmark
framework (the per-dimension, `[0,5]`-scored, `task_name`-routed pipeline
described in the EvalVerseEvaluator "新维度适配指南").

## What this branch provides

| Path | Role |
| --- | --- |
| [`evalverse_evaluator_integration/video_eval/evaluators/arr_evaluator.py`](../evalverse_evaluator_integration/video_eval/evaluators/arr_evaluator.py) | Drop-in `ARRPointwiseEvaluator(BaseEvaluator)`. |
| [`evalverse_evaluator_integration/configs/eval_with_arr.yaml`](../evalverse_evaluator_integration/configs/eval_with_arr.yaml) | Sample `eval.yaml` dimensions wired to the evaluator. |
| [`scripts/v_arr_mvp/generate_per_dim_rubrics.py`](../scripts/v_arr_mvp/generate_per_dim_rubrics.py) | Offline rubric generator: one rubric text file per EvalVerseEvaluator dimension. |

The evaluator wraps the ARR `Judger` in pointwise mode. It is rubric-bound: each
EvalVerseEvaluator dimension is a separate instance of the same class, each
pointing at its own pre-generated rubric file.

## Recipe

### 1. Generate per-dimension rubric files (one-time, offline)

```bash
export OPENAI_API_KEY=<your_api_key>

# Use an EvalVerse pointwise seed (per-dim human_dims expanded). Example:
python scripts/v_arr_mvp/generate_per_dim_rubrics.py \
  --seed_dataset examples/evalverse_raw_pointwise.json \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml \
  --output_dir rubric_pipeline/rubrics

# Alternatively, generate one rubric over the full RewardVerse MVP seed (single
# "overall" dim, suitable for the `v_arr_overall` EvalVerseEvaluator dimension):
python scripts/v_arr_mvp/generate_per_dim_rubrics.py \
  --seed_dataset examples/v_arr_mvp_seed_pairwise.json \
  --config_path rubric_pipeline/config/v_arr_mvp_t2v.yaml \
  --output_dir rubric_pipeline/rubrics \
  --single_dimension v_arr_overall
```

The script reads the seed, runs ARR's generate-verify-refine loop in **pointwise
mode**, and writes one `rubric_pipeline/rubrics/<dimension>.txt` per dimension
present in the seed.

### 2. Wire ARR into EvalVerseEvaluator

```bash
# a. Copy the evaluator class into the framework.
cp evalverse_evaluator_integration/video_eval/evaluators/arr_evaluator.py \
   /path/to/EvalVerseEvaluator/video_eval/evaluators/

# b. Register it via __init__ (one line):
#    EvalVerseEvaluator/video_eval/evaluators/__init__.py
#       from .arr_evaluator import ARRPointwiseEvaluator

# c. Merge the dimension entries from `eval_with_arr.yaml` into
#    EvalVerseEvaluator/configs/eval.yaml.

# d. Tell the evaluator where ARR lives + which rubric directory to use:
export ARR_REPO_ROOT=/abs/path/to/ARR
export ARR_CONFIG_PATH=/abs/path/to/ARR/rubric_pipeline/config/v_arr_mvp_t2v.yaml
export ARR_RUBRICS_DIR=/abs/path/to/ARR/rubric_pipeline/rubrics

# e. Run EvalVerseEvaluator.
bash /path/to/EvalVerseEvaluator/scripts/run_eval.sh
```

### 3. Verify a single dimension

You can drive the evaluator directly without the full pipeline; this mirrors
the framework's `run_single_dimension_xx.sh` style:

```python
from video_eval.data.benchmark_loader import VideoData
from video_eval.evaluators.arr_evaluator import ARRPointwiseEvaluator

ev = ARRPointwiseEvaluator(
    dimension_name="v_arr_overall",
    arr_config_path="/abs/path/to/ARR/rubric_pipeline/config/v_arr_mvp_t2v.yaml",
    arr_rubrics_file="/abs/path/to/ARR/rubric_pipeline/rubrics/v_arr_overall.txt",
    arr_repo_root="/abs/path/to/ARR",
)

vd = VideoData(
    video_id="smoke_001",
    video_path="/abs/path/to/test.mp4",
    category="smoke",
    prompt="A cat walks across a sunlit kitchen floor.",
    metadata={"task_name": "t2v"},
)
print(ev.evaluate(vd))
```

## Task routing (handled by `_build_arr_sample`)

| `task_name` in `VideoData.metadata` | ARR `task_type` | Channels attached |
| --- | --- | --- |
| `t2v` | `t2v` | prompt + video |
| `subject_r2v` | `r2v` | prompt + video + `subject_image` (from `ref_image_paths[0]`) + optional `ground_truth_video` |
| `scene_r2v` | `r2v` | prompt + video + `scene_image` |
| `object_r2v` | `r2v` | prompt + video + `object_image` |
| `camera_r2v` | `r2v` | prompt + video + `ground_truth_video` (from `ref_video_path`) |
| (other / missing) | `t2v` | prompt + video |

If you need a different mapping, override `_build_arr_sample` in a subclass.

## Score normalisation

ARR's pointwise grader returns an integer 1..5. EvalVerseEvaluator expects
`[0.0, 5.0]` with 5 = best.

- Default `score_range="1-5"` clamps the raw ARR score to `[0, 5]` (1..5 already
  fits inside the target range).
- `score_range="0-5"` applies the linear remap `(arr - 1) / 4 * 5`, sending
  1 → 0 and 5 → 5 (useful when a downstream aggregator wants the full range).

## Failure modes

- **Missing rubric file** → returns `score=None`, `details.skipped=True`.
- **VLM error / network failure** → returns `score=0.0`, `details.error=<msg>`.
- **Missing `video_path`** → returns `score=None`, `details.skipped=True`.

## Caveats

- The framework's shared `GeminiApi` instance is **not reused**; ARR constructs
  its own OpenAI-compatible client per `arr_config_path`. Point both at the
  same endpoint to share upstream quota.
- ARR `Judger.initialize()` is called lazily on first `evaluate(...)`, so the
  first evaluation per dimension is slower.
- Rubric files are loaded synchronously; large rubric files inflate every
  prompt. Keep them concise (the ARR generator yields ~5-7 rubrics per dim by
  default).

## Visual Quality MVP (PLCC + SRCC over a small sample)

Sanity-check the pipeline on **one** EvalVerse second-level dimension with a
few dozen samples, and pick the best video input strategy. By default the
driver learns **only the `visual_quality` rubric** (cheapest path); the
remaining 17 dimensions can be learned later when you scale up.

| Script | Purpose |
| --- | --- |
| [`scripts/v_arr_mvp/run_visual_quality_mvp.sh`](../scripts/v_arr_mvp/run_visual_quality_mvp.sh) | End-to-end driver: rubric learning → Visual Quality MVP report. |
| [`scripts/v_arr_mvp/visual_quality_mvp_evaluation.py`](../scripts/v_arr_mvp/visual_quality_mvp_evaluation.py) | Per-strategy PLCC + SRCC comparison; emits JSON + Markdown. |
| [`examples/evalverse_visual_quality_seed.json`](../examples/evalverse_visual_quality_seed.json) | Tiny shape example. Replace with your real EvalVerse Visual Quality records (raw `human_dims` or canonical pointwise are both accepted). |

### Rubric scope knob (`LEARN_DIMENSIONS`)

| `LEARN_DIMENSIONS` value | What gets learned |
| --- | --- |
| `visual_quality` (default) | Only the `visual_quality` rubric is generated; all other dims are skipped even if present in the seed. |
| `all` | Every dimension found in the seed (typically the 18 EvalVerse second-level dims). |
| `visual_quality,chromaticity,lighting` | Comma-separated explicit subset. |

The driver forwards this to `generate_per_dim_rubrics.py --dimensions ...`. When `LEARN_DIMENSIONS=all`, the `--dimensions` flag is omitted and the script picks up every dimension present in the seed.

### Strategies compared by default

| Label | `video_input_strategy` | Frames (`ARR_VIDEO_FRAME_COUNT`) |
| --- | --- | --- |
| `native_video` | `native_video` | n/a (whole clip via `video_url`) |
| `frames_4` | `uniform_frames` | 4 |
| `frames_8` | `uniform_frames` | 8 |
| `frames_16` | `uniform_frames` | 16 |

The driver patches a temp YAML per strategy (`video_input_strategy` +
`eval_mode: pointwise`) and toggles `ARR_VIDEO_FRAME_COUNT` for the
`uniform_frames` variants. Add or remove labels via `--strategies`.

### Metrics

- **PLCC**: Pearson linear correlation between the ARR predicted score and the
  human `label_score`. Captures absolute agreement.
- **SRCC**: Spearman rank correlation. Captures monotonic agreement; robust to
  small calibration offsets.
- **MAE**: mean absolute error in the 1-5 score space.
- **parseable**: number of samples for which ARR returned a usable score.

Higher PLCC/SRCC is better; lower MAE is better.

### Run

```bash
export OPENAI_API_KEY=<your_key>

# MVP default: learn ONLY the visual_quality rubric, then compare strategies.
LEARNING_SEED=examples/your_evalverse_pointwise_train.json \
TEST_DATASET=examples/your_evalverse_visual_quality_test.json \
SAMPLE_LIMIT=30 \
BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
CONFIG_PATH=rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml \
bash scripts/v_arr_mvp/run_visual_quality_mvp.sh

# Same driver, scaled up to learn the 18 EvalVerse second-level dims:
LEARN_DIMENSIONS=all \
LEARNING_SEED=examples/your_evalverse_pointwise_train.json \
bash scripts/v_arr_mvp/run_visual_quality_mvp.sh

# Or invoke the comparator directly against a pre-learned rubric file
# (skip step 1 entirely if you already have rubric_pipeline/rubrics/visual_quality.txt):
python scripts/v_arr_mvp/visual_quality_mvp_evaluation.py \
  --test_dataset examples/your_evalverse_visual_quality_test.json \
  --rubrics_file rubric_pipeline/rubrics/visual_quality.txt \
  --strategies native_video,frames_4,frames_8,frames_16 \
  --limit 30 \
  --output_dir runs/v_arr_mvp/visual_quality
```

The reports land at:

- `runs/v_arr_mvp/visual_quality/visual_quality_strategy_comparison.json`
- `runs/v_arr_mvp/visual_quality/visual_quality_strategy_comparison.md`

The Markdown table flags `Best PLCC` and `Best SRCC`, which is the strategy you
should use when wiring `ARRPointwiseEvaluator` into EvalVerseEvaluator.

### Notes

- The script subsamples with `--limit N` (default 30) and a fixed `--seed` for
  reproducibility. The user explicitly asked for a *small* subset, not the full
  test set, since each strategy means one VLM call per video.
- `scipy.stats.pearsonr` / `spearmanr` are used when available. When scipy is
  missing the script falls back to a pure-Python implementation (verified
  identical on the smoke tests).
- ARR's pointwise grader returns an integer 1..5. PLCC/SRCC are computed on
  the raw 1-5 score (matching EvalVerse's expert annotation scale).

## Visual Quality MVP — rubric input ablation

A second MVP ablation studies **which rubric-generation input format** wins on
Visual Quality. Four variants are learned from the same synthesized pairwise
seed, then compared head-to-head by PLCC + SRCC on the test set with the eval
strategy held fixed (default `native_video`).

| Tag | Inputs to rubric generator | Generator mode | task_description override |
| --- | --- | --- | --- |
| `v1_prompt_ywin` | Prompt + y⁺ | pointwise | default (EvalVerse) |
| `v2_prompt_pair` | Prompt + y⁺ + y⁻ | pairwise | default (EvalVerse) |
| `v3_ywin_only` | y⁺ only (no prompt) | pointwise | "focus on intrinsic visual quality; prompt omitted" |
| `v4_pair_only` | y⁺ + y⁻ (no prompt) | pairwise | "rank purely on intrinsic visual quality" |

### Files

| Path | Role |
| --- | --- |
| [`scripts/v_arr_mvp/synthesize_visual_quality_pairs.py`](../scripts/v_arr_mvp/synthesize_visual_quality_pairs.py) | Mines (y⁺, y⁻) pairs from a per-video score JSONL by grouping on prompt and selecting top-vs-bottom Visual Quality. Emits ARR canonical pairwise JSON + a `.stats.json`. |
| [`scripts/v_arr_mvp/learn_visual_quality_rubric_variants.py`](../scripts/v_arr_mvp/learn_visual_quality_rubric_variants.py) | Builds the four seeds (pointwise y⁺ or pairwise, with/without prompt) and drives ARR's generate-verify-refine loop once per variant. Persists `visual_quality_<tag>.txt` rubric files. |
| [`scripts/v_arr_mvp/visual_quality_rubric_variant_compare.py`](../scripts/v_arr_mvp/visual_quality_rubric_variant_compare.py) | Loads each rubric in turn, scores the (subsampled) test set, and writes a Markdown table flagging Best PLCC + Best SRCC. |
| [`scripts/v_arr_mvp/run_visual_quality_rubric_variants_mvp.sh`](../scripts/v_arr_mvp/run_visual_quality_rubric_variants_mvp.sh) | End-to-end driver: synthesise → learn 4 rubrics → compare. |
| [`examples/merged_output_slim_with_new_path_sample.jsonl`](../examples/merged_output_slim_with_new_path_sample.jsonl) | 8-line shape example of the upstream JSONL (Visual Quality scored, multiple models per prompt). |

### Pair synthesis modes

The synthesiser auto-detects the upstream schema (`--mode auto`, default) and
picks the right mode. Three modes are supported.

#### Mode A: `mos_aesthetics_vq` (default for `merged_evaluate_results_with_path_part*.jsonl`)

The current EvalVerse human-annotation schema. Each line contains a prompt + a
list of generated videos with **per-sub-dim** `average_score` values:

```json
{"path": "<ref>.mp4", "task_name": "t2v", "instruction_prompt": "...",
 "evaluate_data": [
   {"method": "ltx2", "cos_url": "...", "path": "...",
    "evaluate_result": {
      "Aesthetics-Visual_Quality-Physics":              {"average_score": 0.96, ...},
      "Aesthetics-Visual_Quality-Rendering_Quality":    {"average_score": 0.90, ...},
      "Aesthetics-Visual_Quality-Temporal_Consistency": {"average_score": 0.70, ...},
      ...
    }}, ...]}
```

For each video the synthesiser computes the **Visual Quality MoS** as the mean
of the three `Aesthetics-Visual_Quality-*` `average_score` values (skipping
nulls). Within each record the best- and worst-MoS methods become Vw and Vl,
gated by `--min_score_gap` (default `0.10` on the 0-1 scale). Up to
`--max_pairs_per_record` (default `2`) pairs per record are emitted.

If `evaluate_data[*].path` is empty (typical for the MoS file), the synthesiser
falls back to `evaluate_data[*].cos_url`, which ARR's media layer can consume
directly as an HTTP video.

#### Mode B: `method_rank` (legacy, no-MoS EvalVerse corpus)

Used when each line wraps the prompt + a list of generated videos by different
methods (the actual layout of `merged_output_slim_with_new_path.jsonl`):

```json
{"input":  {"instruction_prompt": "...", "task_name": "t2v", "ref_type": "...",
            "path": "...ref.mp4"},
 "output": [{"method": "seedance-2.0", "path": "...", "cos_url": "..."},
            {"method": "wan2.2_14b",   "path": "...", "cos_url": "..."}, ...]}
```

The pairs come from the EvalVerse Tab. 3 Visual Quality **method win-ratio table**:

| Method (normalised) | VQ win-ratio | Method (normalised) | VQ win-ratio |
| --- | --- | --- | --- |
| `seedance-2.0` | 0.71 | `hunyuan1.5` | 0.53 |
| `happy_horse_1.0` | 0.68 | `holocine_full` | 0.44 |
| `ltx2` | 0.67 | `wan2.2_14b` | 0.33 |
| `kling-v3-omni` | 0.66 | `viduq2-pro` | 0.31 |
| `hailuo-2.3` | 0.53 | `univideo` | 0.28 |
| | | `multishot_master_14b` | 0.05 |

For each record the highest- and lowest-ranked methods (`max_pairs_per_record`
top/bottom pairs) become Vw and Vl, gated by `--min_method_gap` (default
`0.20`). Smoke-validated on the real 3102-line dataset (`t2v` only, gap≥0.20)
→ 980 usable records, capped to 200 pairs.

#### Mode C: `score` (flat schema, per-video human scores)

For datasets where each line is one video with a Visual Quality score:

```json
{"prompt": "...", "video_path": "...",
 "human_dims": {"Visual Quality": 4}}
```

| What | Accepted keys (first non-empty wins) |
| --- | --- |
| prompt | `prompt` / `caption` / `text` / `query` / `prompt_text` / `instruction_prompt` |
| video path | `new_path` / `video_path` / `video` / `path` / `url` |
| Visual Quality score | `visual_quality` / `vq_score` / `quality_score`, or `human_dims["Visual Quality"]` (also `visual_quality`, `visual quality`, `VQ`) |

Pairs are emitted only when the score gap is `≥ --min_score_gap` (default 1.0
on the 1-5 scale).

### Run

```bash
export OPENAI_API_KEY=<your_key>

# Default: synthesise 30 pairs from the MoS-annotated JSONL.
INPUT_JSONL=merged_evaluate_results_with_path_part1.jsonl \
TEST_DATASET=examples/your_evalverse_visual_quality_test.json \
SAMPLE_LIMIT=30 \
STRATEGY=native_video \
bash scripts/v_arr_mvp/run_visual_quality_rubric_variants_mvp.sh

# Older method-rank corpus (no MoS, uses paper win-ratios):
INPUT_JSONL=merged_output_slim_with_new_path.jsonl \
SYNTH_MODE=method_rank \
MIN_SCORE_GAP=0.20 \
MAX_PAIRS=200 \
bash scripts/v_arr_mvp/run_visual_quality_rubric_variants_mvp.sh

# Reports land at:
#   runs/v_arr_mvp/visual_quality_rubric_variants/
#     synthesis.log
#     rubric_learning.log
#     visual_quality_rubric_variant_comparison.json
#     visual_quality_rubric_variant_comparison.md
```

### Smoke validated

On the real EvalVerse JSONL (3102 records, mode=`method_rank`,
task_name=`t2v`, gap=0.20):

```text
loaded                      : 3102
nested_records              : 3102
skipped_wrong_task          : 2096   (R2V records held back)
skipped_lt_two_methods      : 7
skipped_unknown_methods     : 0
skipped_no_gap              : 19     (top vs bottom method gap < 0.20)
records_used                : 980
final_pairs                 : 200    (capped by --max_total_pairs)
```

Typical pairs:

```text
seedance-2.0 (0.71) > wan2.2_14b (0.33)   gap=0.38
ltx2 (0.67)         > wan2.2_14b (0.33)   gap=0.34
kling-v3-omni(0.66) > univideo  (0.28)    gap=0.38
```

The 3-record `examples/merged_output_slim_with_new_path_sample.jsonl` exercises
the same code path on a minimum-sized fixture.
