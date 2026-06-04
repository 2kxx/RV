# RewardVerse MVP

A parallel, generic Auto-Rubric pipeline on top of the EvalVerse video stack. The
MVP answers four research questions before any video RPO training begins:

| RQ | Question | Pass threshold |
| --- | --- | --- |
| RQ1 | Can a Gemini-driven Auto-Rubric judge recover pairwise human preferences? | accuracy ≥ 70 % |
| RQ2 | Do **temporal** rubrics emerge without prescribing dimension names? | temporal share ≥ 30 % |
| RQ3 | Is the judge robust to Video1 / Video2 position swap? | consistency ≥ 70 % |
| RQ4 | Does the rubric judge beat a zero-shot Gemini judge with no rubric? | Δ ≥ +5 percentage points |

For the standalone EvalVerse **Visual Quality** dimension MVP (rubric learning
over 18 second-level dims + PLCC/SRCC comparison across video input
strategies), see [`docs/evalverse_evaluator_integration.md`](evalverse_evaluator_integration.md#visual-quality-mvp-plcc--srcc-over-a-small-sample).

## Files added

| Path | Role |
| --- | --- |
| [`rubric_pipeline/config/v_arr_mvp_t2v.yaml`](../rubric_pipeline/config/v_arr_mvp_t2v.yaml) | Gemini-driven, generic T2V config. Forces spatial + temporal rubric layers, no EvalVerse vocabulary. |
| [`rubric_pipeline/datasets/generic_pairwise_adapter.py`](../rubric_pipeline/datasets/generic_pairwise_adapter.py) | Convert GenAI-Bench-Video / VideoGen-RewardBench / generic pairwise JSON to ARR canonical form. |
| [`examples/v_arr_mvp_seed_pairwise.json`](../examples/v_arr_mvp_seed_pairwise.json) | Placeholder seed (replace with 200 mined pairs). |
| [`examples/v_arr_mvp_test_pairwise.json`](../examples/v_arr_mvp_test_pairwise.json) | Placeholder held-out test (replace with 100 pairs). |
| [`scripts/v_arr_mvp/classify_rubric_categories.py`](../scripts/v_arr_mvp/classify_rubric_categories.py) | Reclassify rubrics into {spatial, temporal, instruction, quality}. |
| [`scripts/v_arr_mvp/zero_shot_judge.py`](../scripts/v_arr_mvp/zero_shot_judge.py) | Baseline pairwise judge without rubric. |
| [`scripts/v_arr_mvp/positional_bias.py`](../scripts/v_arr_mvp/positional_bias.py) | Swap-consistency diagnostic. |
| [`scripts/v_arr_mvp/build_mvp_report.py`](../scripts/v_arr_mvp/build_mvp_report.py) | Aggregate the four RQ metrics into a Markdown report. |
| [`scripts/v_arr_mvp/run_mvp_pipeline.sh`](../scripts/v_arr_mvp/run_mvp_pipeline.sh) | End-to-end driver. |

## Quick start

```bash
export OPENAI_API_KEY=<your_gemini_api_key>

# 1. Run the full pipeline (uses the seed/test placeholders by default).
bash scripts/v_arr_mvp/run_mvp_pipeline.sh

# 2. Inspect the decision sheet.
cat runs/v_arr_mvp/mvp_report.md
```

## Design notes

- **No EvalVerse vocabulary**: `v_arr_mvp_t2v.yaml`'s `task_description` is injected near
  the top of every rubric prompt and explicitly tells the VLM to ignore any earlier
  EvalVerse `[Aspect > Main > Sub]` framing. Rubrics must be organised into a SPATIAL
  layer and a TEMPORAL layer, with at least one third in the temporal layer.
- **Gemini transport**: Gemini's OpenAI-compatible endpoint
  (`https://generativelanguage.googleapis.com/v1beta/openai/`) plugs into the existing
  [`OpenAIChatModel`](../rubric_pipeline/models/openai_chat_model.py) without code changes.
  `video_input_strategy: uniform_frames` converts each clip to base64 image frames,
  which is the most portable channel through that endpoint.
- **No modifications to EvalVerse code paths**: `evalverse_adapter.py`, the EvalVerse
  YAMLs, and the EvalVerse-tagged T2V prompts are untouched. MVP files are strictly
  additive.
- **Decision tree**: see the bottom of the generated report. `GREEN` = proceed to
  Paradigm C; `YELLOW` = harden the prompt and retry; `RED` = fall back to RewardVerse as
  evaluator only.
