# RewardVerse — 18-dimension pointwise test sets

This index lists where each EvalVerse second-level dimension's pointwise
**test set** lives, so downstream evaluation drivers (the rubric pointwise
judger and any baseline runners) can find the right file without
hard-coding paths.

## Schema

Every test JSON is a list of records with the same shape as
`data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json`:

```jsonc
{
  "id":          "varr_<dim>_test_NNN",
  "query":       "<English prompt or '(no prompt supplied; ...)'>",
  "response":    "https://.../<video>.mp4",
  "label_score": 1.0–5.0,                  // = 1 + 4 * mean(EvalVerse sub-keys)
  "dimension":   "<dim>",
  "task_type":   "t2v" | "r2v",
  "metadata":    { ...EvalVerse provenance... }
}
```

`label_score = 1 + 4 * mean(sub_scores)` — same mapping as `_to_label_score`
in `build_visual_quality_mvp_dataset.py`, so T1 and T2 sit on a common scale.

## File index

| Dimension | File | Count | Label range | Notes |
|---|---|---|---|---|
| character | `data/v_arr_mvp/character_mvp/test_pointwise_character.json` | 30 | 1.22–5.00 | T2 |
| scene | `data/v_arr_mvp/scene_mvp/test_pointwise_scene.json` | 30 | 1.68–5.00 | T2 |
| consistency | `data/v_arr_mvp/consistency_mvp/test_pointwise_consistency.json` | 30 | 1.40–4.88 | T2 |
| action | `data/v_arr_mvp/action_mvp/test_pointwise_action.json` | 30 | 2.43–4.93 | T2 |
| expression | `data/v_arr_mvp/expression_mvp/test_pointwise_expression.json` | 30 | 3.29–5.00 | T2; data ceiling |
| composition | `data/v_arr_mvp/composition_mvp/test_pointwise_composition.json` | 30 | 2.36–5.00 | T2 |
| lens | `data/v_arr_mvp/lens_mvp/test_pointwise_lens.json` | 30 | 3.12–5.00 | T2; data ceiling |
| pacing | `data/v_arr_mvp/pacing_mvp/test_pointwise_pacing.json` | 30 | 1.00–4.92 | T2 |
| chromaticity | `data/v_arr_mvp/chromaticity_mvp/test_pointwise_chromaticity.json` | 30 | 3.36–5.00 | T2; data ceiling |
| materiality | `data/v_arr_mvp/materiality_mvp/test_pointwise_materiality.json` | 30 | 1.38–5.00 | T2 |
| lighting | `data/v_arr_mvp/lighting_mvp/test_pointwise_lighting.json` | 30 | 1.62–5.00 | T2 |
| **visual_quality** | `data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json` | 30 | 1.04–4.77 | **T1** (re-uses the existing VQ MVP shared test set; full 5-band coverage) |
| grounding | `data/v_arr_mvp/grounding_mvp/test_pointwise_grounding.json` | 30 | 1.00–4.68 | T2 |
| progression | `data/v_arr_mvp/progression_mvp/test_pointwise_progression.json` | 30 | 1.00–5.00 | T2 |
| logic | `data/v_arr_mvp/logic_mvp/test_pointwise_logic.json` | 19 | 3.57–5.00 | T2; **source data ceiling** — only 19 records have ≥1 entry with all 4 logic sub-keys scored. |
| rhythm | `data/v_arr_mvp/rhythm_mvp/test_pointwise_rhythm.json` | 30 | 3.68–5.00 | T2; data ceiling |
| vocal | `data/v_arr_mvp/vocal_mvp/test_pointwise_vocal.json` | 30 | 3.34–5.00 | T2; needs audio (Gemini multimodal_url native path carries it) |
| soundscape | `data/v_arr_mvp/soundscape_mvp/test_pointwise_soundscape.json` | 30 | 2.89–5.00 | T2; needs audio |

**Total**: 529 pointwise samples across 18 dimensions.

## Generation

Re-generate any single dim:

```bash
python scripts/v_arr_mvp/build_dim_seed_v2.py \
    --dimension <dim> \
    --mode test_pointwise \
    --target_count 30 \
    --exclude_pair_dataset data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json
```

Re-generate all 17 (visual_quality is fixed by `shared_test_pointwise.json`,
do **not** regenerate it via `--mode test_pointwise`):

```bash
for dim in character scene consistency action expression composition lens \
          pacing chromaticity materiality lighting grounding progression \
          logic rhythm vocal soundscape; do
    python scripts/v_arr_mvp/build_dim_seed_v2.py \
        --dimension $dim --mode test_pointwise --target_count 30 \
        --exclude_pair_dataset data/v_arr_mvp/${dim}_mvp/v2_prompt_pair.json
done
```

If new Chinese prompts surface as cache misses, drop them into
`scripts/v_arr_mvp/_pending_translations_t2.json`, hand-translate, and merge
into the shared cache via
`python scripts/v_arr_mvp/_inject_t2_translations.py`.

## Disjointness invariants

For each dimension `D` (other than visual_quality) the test set is built with
`--exclude_pair_dataset data/v_arr_mvp/<D>_mvp/v2_prompt_pair.json`, so the
following property holds by construction:

```
test_pointwise_D.json[*].response  ∩  v2_prompt_pair_D.json[*].response[*]  =  ∅
```

This means the rubric-generation training pairs (`v2_prompt_pair_D.json`) and
the per-dim test set (`test_pointwise_D.json`) share **zero video URLs**, so
PLCC/SRCC computed on the test set is not contaminated by training-data
memorization at the URL level.

For visual_quality, the analogous invariant is enforced by the existing
`build_visual_quality_mvp_dataset.py` pipeline: `shared_test_pointwise.json`
URLs are excluded from `v1_prompt_ywin.json` / `v3_ywin_only.json`, and may
overlap (only on the loser side) with `v2_prompt_pair.json` / `v4_pair_only.json`
(per the A1 design choice in [docs/v_arr_mvp.md](v_arr_mvp.md)).

## External benchmarks (cross-dataset SOTA comparison)

For the SOTA table we additionally evaluate on **one** external benchmark
that is NOT EvalVerse-derived. Pulled by
[`scripts/v_arr_mvp/fetch_external_benchmarks.py`](../scripts/v_arr_mvp/fetch_external_benchmarks.py)
from the local HF snapshot under
`/group/40063/jacuzzitang/datasets/VideoGen-RewardBench/`.

| Benchmark | Mode | File | Default subset | Generator subcommand |
|---|---|---|---|---|
| **VideoGen-RewardBench** (Kling/KwaiVGI) | pairwise | `vgrb_pairwise_<N>.json` | **FULL** eval split (~25.2k pairs) | `videogen_rewardbench` |

> The earlier 200-pair stratified subset has been retired in favour of the
> full eval split (see [`runs/STATUS_SYNC.md`](../runs/STATUS_SYNC.md)).

### Generation

```bash
# VideoGen-RewardBench pairwise (FULL eval split, default)
python scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench

# Smoke / dev: stratified 200-pair sample
python scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench \
    --num_pairs 200 --seed 0
```

The emitted JSON shares its schema with the internal `v2_prompt_pair.json`
(listwise pairwise) — consumable by the pairwise path in `judger.py`
(and by [`rubric_pipeline/datasets/generic_pairwise_adapter.py`](../rubric_pipeline/datasets/generic_pairwise_adapter.py)).

### Per-benchmark dimension routing (D6)

When the SOTA-table builder runs ours / baselines on these benchmarks, it
only invokes the **relevant** ARR dimensions instead of all 18, because the
external benchmarks only expose 4-axis labels. The mapping is:

| External axis | ARR L2 dims to invoke | ARR L1 group to compare against |
|---|---|---|
| Visual Quality | visual_quality, chromaticity, lighting, materiality | aesthetics |
| Temporal Consistency | consistency, pacing | acting + cinematography mean |
| Motion Quality | action, expression, pacing | acting + cinematography mean |
| Text Alignment | character, scene, composition | visual_concept_design + cinematography mean |
| Overall | mean of all 7 L1 groups | top-level mean |

This keeps the API cost per external benchmark at ~3-5 ARR L2 calls per
sample instead of all 18, per user decision (D6).
