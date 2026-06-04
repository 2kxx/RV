# Data Formats

This page describes the JSON shapes accepted by `rubric_pipeline/utils/vision.py` and
`rubric_pipeline/utils/media.py`.
For rubric generation strategy, VLM choice, and reuse, start from the
[Auto-Rubric Guide](auto_rubric/README.md).

## Pairwise Text-To-Image

Use this format for FLUX rubric generation and evaluation:

```json
{
  "query": "A cinematic portrait of a ceramic robot chef.",
  "response": ["path/to/image_a.png", "path/to/image_b.png"],
  "label_rank": [1, 2]
}
```

`label_rank` uses smaller-is-better rank numbers. `[1, 2]` means the first image is preferred.

Accepted image keys for text-to-image include:

- `response`
- `responses`
- `images`
- `image_paths`
- `generated_images`

## Pairwise Image Editing

Use this format for Qwen-Image-Edit:

```json
{
  "query": "Replace the sky with a sunset while preserving the building.",
  "source_image": "path/to/source.png",
  "edited_images": ["path/to/edit_a.png", "path/to/edit_b.png"],
  "label_rank": [2, 1]
}
```

Accepted source image keys include:

- `source_image`
- `base_image`
- `original_image`
- `input_image`
- `reference_image`

Accepted edited-output keys include:

- `edited_images`
- `target_images`
- `output_images`
- `result_images`
- `response`

## EvalVerse Video Tracks (SIGGRAPH Asia 2026)

EvalVerse covers four video generation tracks. Each is mapped to an Auto-Rubric `task_type`:

| EvalVerse task | ARR `task_type` | Config |
| --- | --- | --- |
| Text-to-Video | `t2v` | [`qwen3vl_8B_instruct_t2v.yaml`](../rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml) |
| Reference-to-Video (Subject / Object / Scene / Camera) | `r2v` | [`qwen3vl_8B_instruct_r2v.yaml`](../rubric_pipeline/config/qwen3vl_8B_instruct_r2v.yaml) |
| Text-to-Multi-Shot-Video | `t2v_multishot` | [`qwen3vl_8B_instruct_multishot.yaml`](../rubric_pipeline/config/qwen3vl_8B_instruct_multishot.yaml) |
| Text-to-Video-with-Sound | `t2v_sound` | [`qwen3vl_8B_instruct_video_sound.yaml`](../rubric_pipeline/config/qwen3vl_8B_instruct_video_sound.yaml) |

EvalVerse rubrics follow the pipeline-aware taxonomy:

```
3 stages (Pre / Production / Post) -> 7 aspects -> 18 main dims -> 45 sub dims -> 196 rationales
```

Aspects: Visual Concept Design, Acting, Cinematography, Aesthetics, Affectivity,
Multi-Shot, Sound Design.

### Pairwise (Bradley-Terry preference)

EvalVerse Stage 1 fine-tuning uses Win/Lose pairs. Either shape is accepted:

```json
{
  "id": "ev_t2v_001",
  "task": "text_to_video",
  "prompt": "...",
  "win_video": "path/to/Vw.mp4",
  "lose_video": "path/to/Vl.mp4"
}
```

```json
{
  "query": "...",
  "response": ["path/to/Vw.mp4", "path/to/Vl.mp4"],
  "label_rank": [1, 2],
  "task_type": "t2v",
  "metadata": {"evalverse_id": "ev_t2v_001"}
}
```

`label_rank=[1, 2]` corresponds to (Vw, Vl) in Bradley-Terry ranking loss.

### Pointwise (Score Calibration, with CoT rationale)

EvalVerse Stage 2 fine-tuning uses pointwise scores per dimension with optional CoT
rationale `Z`. Accepted upstream shape:

```json
{
  "id": "ev_pw_001",
  "task": "text_to_video",
  "prompt": "...",
  "video": "path/to/video.mp4",
  "human_dims": {
    "Visual Quality": 4,
    "Composition": 5,
    "Lighting": 4
  },
  "rationale": "Lighting and composition are strong; minor temporal flicker."
}
```

The adapter expands a multi-dim record into one ARR sample per dimension.

### Reference-to-Video

```json
{
  "id": "ev_r2v_001",
  "task": "reference_to_video",
  "prompt": "...",
  "ground_truth_video": "path/to/gt.mp4",
  "subject_image": "path/to/subject.png",
  "scene_image": "path/to/scene.png",
  "win_video": "path/to/Vw.mp4",
  "lose_video": "path/to/Vl.mp4"
}
```

R2V reference channels (subject / object / scene / depth_sequence) and the
ground-truth video are forwarded to the VLM and to the EvalVerse Reference-Adherence
sub-dimension under Aesthetics > Materiality.

### Multi-Shot

```json
{
  "id": "ev_ms_001",
  "task": "text_to_multi_shot_video",
  "prompt": "...",
  "shot_sequence": [
    {"shot_size": "wide", "duration_sec": 2.5},
    {"shot_size": "medium", "duration_sec": 3.0}
  ],
  "win_video": "path/to/Vw.mp4",
  "lose_video": "path/to/Vl.mp4"
}
```

### Video-with-Sound

```json
{
  "id": "ev_vs_001",
  "task": "text_to_video_with_sound",
  "prompt": "...",
  "win_video": "path/to/Vw.mp4",
  "lose_video": "path/to/Vl.mp4",
  "audio": "path/to/audio.wav",
  "metadata": {
    "operator_evidence": {
      "lip_sync": {"sync_offset_ms": 35, "confidence": 0.92}
    }
  }
}
```

`metadata.operator_evidence` is the EvalVerse `E_prof` perception prior surface
(DINO / InsightFace / YOLO / SyncNet / Whisper). See
[`rubric_pipeline/operators/__init__.py`](../rubric_pipeline/operators/__init__.py).

### EvalVerse raw conversion

```bash
python scripts/eval/convert_evalverse_dataset.py \
  --input examples/evalverse_raw_pairwise.json \
  --output examples/seed_t2v_pairwise.json
```

Set `ARR_VIDEO_INPUT_STRATEGY=native_video` (default) or `uniform_frames`. Tune
`video_input_strategy` in the YAML configs above.

## Pointwise Mode

Pointwise grading is available, although the included training launchers use pairwise RPO.

```json
{
  "query": "A realistic photo of a glass teapot.",
  "response": "path/to/image.png",
  "label_score": 4
}
```

Set `eval_mode: "pointwise"` and configure `min_score` / `max_score` in the YAML file.

## Path Rules

Paths may be absolute, relative to the current working directory, or URLs/data URLs accepted by the OpenAI-compatible vision endpoint. Local image paths are encoded as base64 image URLs; local videos are encoded as base64 video URLs (or uniformly sampled frames when using `uniform_frames`).

## Related Guides

- [Rubric Design](auto_rubric/rubric_design.md)
- [Rubric Reuse](auto_rubric/rubric_reuse.md)
- [Auto-Rubric Workflows](auto_rubric/workflows.md)
