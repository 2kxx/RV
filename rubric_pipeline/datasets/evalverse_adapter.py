# -*- coding: utf-8 -*-
"""Convert EvalVerse-style annotations into Auto-Rubric canonical JSON.

EvalVerse taxonomy (SIGGRAPH Asia 2026): 3 production stages -> 7 cinematic
aspects -> 18 main dimensions -> 45 sub-dimensions -> 196 granular rationales.

References:
- Pre-Production: Visual Concept Design (Character, Scene).
- Production: Acting, Cinematography, Aesthetics, Affectivity.
- Post-Production: Multi-Shot, Sound Design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

# ============================================================
# EvalVerse Taxonomy
# ============================================================

# Top-level cinematic aspects (7).
EVALVERSE_ASPECTS = (
    "visual_concept_design",
    "acting",
    "cinematography",
    "aesthetics",
    "affectivity",
    "multi_shot",
    "sound_design",
)

# Mapping aspect -> main dimensions (18 in total).
EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "visual_concept_design": ("character", "scene"),
    "acting": ("consistency", "action", "expression"),
    "cinematography": ("composition", "pacing", "lens"),
    "aesthetics": ("visual_quality", "chromaticity", "materiality", "lighting"),
    "affectivity": ("grounding", "progression"),
    "multi_shot": ("logic", "rhythm"),
    "sound_design": ("vocal", "soundscape"),
}

# Mapping main dimension -> sub dimensions (45 total).
EVALVERSE_MAIN_TO_SUB_DIMENSIONS: dict[str, tuple[str, ...]] = {
    # Pre-Production: Visual Concept Design
    "character": ("identification", "costume_rationality"),
    "scene": ("environment_plausibility", "genre_distinctiveness"),
    # Production: Acting
    "consistency": ("face_identity", "attribute"),
    "action": ("action_tension", "action_emotion_synergy", "interaction_plausibility"),
    "expression": ("accuracy", "continuity", "facial_tension", "expression_diversity"),
    # Production: Cinematography
    "composition": ("spatial_layering", "shot_size_rationality", "subject_prominence"),
    "pacing": ("movement_rationality",),
    "lens": ("depth_of_field", "focus", "focal_length", "exposure"),
    # Production: Aesthetics
    "visual_quality": ("temporal_consistency", "rendering_quality", "physics"),
    "chromaticity": ("harmony", "emotive_power"),
    "materiality": ("material_identifiability", "stylistic_consistency", "reference_adherence"),
    "lighting": ("lighting_logic", "volumetric_sculpting"),
    # Production: Affectivity
    "grounding": ("tonal_identifiability", "visual_emotional_synergy"),
    "progression": ("transition_continuity", "intensity_layering"),
    # Post-Production: Multi-Shot
    "logic": ("scene_consistency", "narrative_continuity", "spatial_continuity"),
    "rhythm": ("shot_duration_rationality", "rhythmic_layering"),
    # Post-Production: Sound Design
    "vocal": ("acoustic_quality", "lip_sync", "narrative_sound_design"),
    "soundscape": ("ambient_sound_fidelity", "musical_score_alignment"),
}

# Flat list of canonical names for filtering / scoring.
EVALVERSE_MAIN_DIMENSIONS: tuple[str, ...] = tuple(
    d for dims in EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS.values() for d in dims
)
EVALVERSE_SUB_DIMENSIONS: tuple[str, ...] = tuple(
    d for subs in EVALVERSE_MAIN_TO_SUB_DIMENSIONS.values() for d in subs
)

# Back-compat alias used by earlier code paths.
EVALVERSE_DIMENSIONS = EVALVERSE_ASPECTS

# Alias map for normalizing free-form names from the EvalVerse PDF / human
# annotations into canonical taxonomy keys.
_DIMENSION_ALIASES: dict[str, str] = {
    # aspects
    "visual concept design": "visual_concept_design",
    "visual_concept_design": "visual_concept_design",
    "vcd": "visual_concept_design",
    "acting": "acting",
    "cinematography": "cinematography",
    "aesthetics": "aesthetics",
    "affectivity": "affectivity",
    "multi-shot": "multi_shot",
    "multi_shot": "multi_shot",
    "multishot": "multi_shot",
    "post_production": "multi_shot",
    "sound design": "sound_design",
    "sound_design": "sound_design",
    "audio": "sound_design",
    # main dims
    "character": "character",
    "scene": "scene",
    "consistency": "consistency",
    "action": "action",
    "expression": "expression",
    "composition": "composition",
    "pacing": "pacing",
    "lens": "lens",
    "visual quality": "visual_quality",
    "visual_quality": "visual_quality",
    "chromaticity": "chromaticity",
    "materiality": "materiality",
    "lighting": "lighting",
    "grounding": "grounding",
    "progression": "progression",
    "logic": "logic",
    "rhythm": "rhythm",
    "vocal": "vocal",
    "soundscape": "soundscape",
    # sub dims (common abbreviations seen in EvalVerse tables)
    "identifiability": "identification",
    "costume rationality": "costume_rationality",
    "environment plausibility": "environment_plausibility",
    "genre distinctiveness": "genre_distinctiveness",
    "face identity": "face_identity",
    "attribute": "attribute",
    "action tension": "action_tension",
    "action emotion synergy": "action_emotion_synergy",
    "interaction rationality": "interaction_plausibility",
    "interaction plausibility": "interaction_plausibility",
    "accuracy": "accuracy",
    "continuity": "continuity",
    "facial tension": "facial_tension",
    "expression diversity": "expression_diversity",
    "spatial layering": "spatial_layering",
    "shot size rationality": "shot_size_rationality",
    "shot-size rationality": "shot_size_rationality",
    "subject prominence": "subject_prominence",
    "movement rationality": "movement_rationality",
    "depth of field": "depth_of_field",
    "focus": "focus",
    "focal length": "focal_length",
    "exposure": "exposure",
    "temporal consistency": "temporal_consistency",
    "rendering quality": "rendering_quality",
    "physics": "physics",
    "harmony": "harmony",
    "emotive power": "emotive_power",
    "material identifiability": "material_identifiability",
    "stylistic consistency": "stylistic_consistency",
    "reference adherence": "reference_adherence",
    "lighting logic": "lighting_logic",
    "volumetric sculpting": "volumetric_sculpting",
    "tonal identifiability": "tonal_identifiability",
    "visual emotional synergy": "visual_emotional_synergy",
    "visual-emotional synergy": "visual_emotional_synergy",
    "transition continuity": "transition_continuity",
    "intensity layering": "intensity_layering",
    "scene consistency": "scene_consistency",
    "narrative continuity": "narrative_continuity",
    "spatial continuity": "spatial_continuity",
    "shot duration rationality": "shot_duration_rationality",
    "shot-duration rationality": "shot_duration_rationality",
    "rhythmic layering": "rhythmic_layering",
    "acoustic quality": "acoustic_quality",
    "lip sync": "lip_sync",
    "lip-sync": "lip_sync",
    "lipsync": "lip_sync",
    "narrative sound design": "narrative_sound_design",
    "ambient sound fidelity": "ambient_sound_fidelity",
    "musical score alignment": "musical_score_alignment",
    # legacy short aliases retained for backward compatibility
    "alignment": "visual_concept_design",
    "prompt_adherence": "visual_concept_design",
    "motion": "pacing",
    "motion_quality": "pacing",
    "aesthetic": "aesthetics",
    "aesthetic_fit": "aesthetics",
    "physical_plausibility": "physics",
    "physical plausibility": "physics",
    "text_video_alignment": "visual_concept_design",
    "text-video alignment": "visual_concept_design",
    # "美学质量" and other deployments that promote sub-dims to second-level
    "aesthetic quality": "visual_quality",
    "aesthetic_quality": "visual_quality",
    "美学质量": "visual_quality",
    "美学": "aesthetics",
    # Some EvalVerse exports split Acting/Action sub-dims into a 19th second-level dim
    "interaction": "interaction_plausibility",
    "interaction_rationality": "interaction_plausibility",
}


# ============================================================
# EvalVerse Task Catalogue (Sec. 4 + Fig. 6)
# ============================================================

EVALVERSE_TASK_T2V = "t2v"
EVALVERSE_TASK_R2V = "r2v"
EVALVERSE_TASK_SUBJECT_TO_VIDEO = "subject_to_video"
EVALVERSE_TASK_OBJECT_TO_VIDEO = "object_to_video"
EVALVERSE_TASK_SCENE_TO_VIDEO = "scene_to_video"
EVALVERSE_TASK_CAMERA_TO_VIDEO = "camera_to_video"
EVALVERSE_TASK_MULTISHOT = "text_to_multi_shot_video"
EVALVERSE_TASK_WITH_SOUND = "text_to_video_with_sound"

_TASK_ALIASES: dict[str, str] = {
    "t2v": EVALVERSE_TASK_T2V,
    "text_to_video": EVALVERSE_TASK_T2V,
    "text-to-video": EVALVERSE_TASK_T2V,
    "r2v": EVALVERSE_TASK_R2V,
    "reference_to_video": EVALVERSE_TASK_R2V,
    "reference-to-video": EVALVERSE_TASK_R2V,
    "subject_to_video": EVALVERSE_TASK_SUBJECT_TO_VIDEO,
    "object_to_video": EVALVERSE_TASK_OBJECT_TO_VIDEO,
    "scene_to_video": EVALVERSE_TASK_SCENE_TO_VIDEO,
    "camera_to_video": EVALVERSE_TASK_CAMERA_TO_VIDEO,
    "text_to_multi_shot_video": EVALVERSE_TASK_MULTISHOT,
    "multi_shot": EVALVERSE_TASK_MULTISHOT,
    "multishot": EVALVERSE_TASK_MULTISHOT,
    "text_to_video_with_sound": EVALVERSE_TASK_WITH_SOUND,
    "video_with_sound": EVALVERSE_TASK_WITH_SOUND,
    "audio_visual": EVALVERSE_TASK_WITH_SOUND,
}


def normalize_evalverse_task(task: str | None) -> str:
    if not task:
        return EVALVERSE_TASK_T2V
    key = task.strip().lower().replace("-", "_").replace(" ", "_")
    return _TASK_ALIASES.get(key, key)


# ============================================================
# Helpers
# ============================================================


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _normalize_dimension(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace("/", "_")
    if key in _DIMENSION_ALIASES:
        return _DIMENSION_ALIASES[key]
    spaced = key.replace("_", " ")
    return _DIMENSION_ALIASES.get(spaced, key)


def aspect_of(dimension: str) -> str | None:
    """Return the parent aspect for a canonical dimension name.

    Resolution order:
        1. If the dimension is itself an aspect (e.g. "aesthetics") -> return it.
        2. If it is a main dim (e.g. "visual_quality") -> return its aspect.
        3. If it is a sub dim (e.g. "temporal_consistency") -> walk up to aspect.
    """
    canonical = _normalize_dimension(dimension)
    if canonical in EVALVERSE_ASPECTS:
        return canonical
    for aspect, mains in EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS.items():
        if canonical in mains:
            return aspect
    for main, subs in EVALVERSE_MAIN_TO_SUB_DIMENSIONS.items():
        if canonical in subs:
            for aspect, mains in EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS.items():
                if main in mains:
                    return aspect
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Mapping):
        return [value]
    return list(value)


def _resolve_video_path(record: Mapping[str, Any], candidate: Mapping[str, Any] | str) -> str:
    if isinstance(candidate, str):
        return candidate
    for key in ("path", "video_path", "video", "file", "url"):
        if candidate.get(key):
            return str(candidate[key])
    video_id = _first_non_empty(candidate.get("video_id"), candidate.get("id"))
    root = _first_non_empty(record.get("video_root"), record.get("root"))
    if video_id and root:
        return str(Path(root) / f"{video_id}.mp4")
    raise ValueError(f"Cannot resolve video path from candidate: {candidate}")


def _extract_reference_assets(record: Mapping[str, Any]) -> dict[str, Any]:
    """Pull EvalVerse Reference-to-Video / Camera-to-Video assets.

    Reference channels (Sec. 4, Fig. 6):
    - subject reference image (Subject-to-Video)
    - object reference image (Object-to-Video)
    - scene reference image (Scene-to-Video)
    - depth sequence (Camera-to-Video)
    - ground truth video (always present for R2V)
    """
    refs: dict[str, Any] = {}
    for source_key, target_key in (
        ("subject", "subject_image"),
        ("subject_image", "subject_image"),
        ("subject_reference", "subject_image"),
        ("object", "object_image"),
        ("object_image", "object_image"),
        ("scene", "scene_image"),
        ("scene_image", "scene_image"),
        ("depth", "depth_sequence"),
        ("depth_sequence", "depth_sequence"),
        ("control", "depth_sequence"),
        ("ground_truth_video", "ground_truth_video"),
        ("gt_video", "ground_truth_video"),
        ("reference_video", "ground_truth_video"),
    ):
        value = record.get(source_key)
        if value:
            refs[target_key] = str(value) if not isinstance(value, list) else [str(v) for v in value]
    audio = record.get("audio") or record.get("audio_path") or record.get("audio_track")
    if audio:
        refs["audio"] = str(audio) if not isinstance(audio, list) else [str(a) for a in audio]
    multi_shot = record.get("shot_sequence") or record.get("shots")
    if multi_shot:
        refs["shot_sequence"] = multi_shot
    return refs


def _build_metadata(record: Mapping[str, Any], task: str) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    sample_id = _first_non_empty(
        record.get("id"),
        record.get("sample_id"),
        record.get("evalverse_id"),
    )
    if sample_id:
        metadata.setdefault("evalverse_id", sample_id)
    if record.get("split"):
        metadata["split"] = record["split"]
    metadata["evalverse_task"] = task
    for opt_key in ("annotator", "expert_panel", "stage", "rationale"):
        if record.get(opt_key):
            metadata[opt_key] = record[opt_key]
    return metadata


def _normalize_dim_scores(dim_scores: Mapping[str, Any]) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for k, v in dim_scores.items():
        canon = _normalize_dimension(str(k))
        if v is None:
            continue
        try:
            out[canon] = float(v)
        except (TypeError, ValueError):
            continue
    return out


# ============================================================
# Conversion
# ============================================================


def convert_evalverse_record(record: Mapping[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert one EvalVerse record to ARR canonical sample(s).

    Supported shapes:
    - Preference Alignment: pairwise win/lose video pair (Sec. 6.2.1).
    - Score Calibration: pointwise per-dimension scores with optional CoT
      rationale (Sec. 6.2.2).
    - Reference-to-Video / Camera-to-Video / Multi-Shot / Video-with-Sound.
    """
    query = _first_non_empty(
        record.get("query"),
        record.get("prompt"),
        record.get("caption"),
        record.get("text"),
    )
    task = normalize_evalverse_task(
        _first_non_empty(record.get("task"), record.get("evalverse_task"), record.get("task_type")),
    )
    metadata = _build_metadata(record, task)
    references = _extract_reference_assets(record)
    if references:
        metadata["references"] = references

    # ---- Bradley-Terry pairwise (Vw, Vl) ----
    win_video = _first_non_empty(record.get("win_video"), record.get("Vw"), record.get("preferred"))
    lose_video = _first_non_empty(record.get("lose_video"), record.get("Vl"), record.get("rejected"))
    if win_video and lose_video:
        return {
            "query": query,
            "response": [str(win_video), str(lose_video)],
            "label_rank": [1, 2],
            "task_type": _map_to_arr_task_type(task),
            "metadata": metadata,
            **{k: v for k, v in references.items() if k in {"subject_image", "object_image", "scene_image", "depth_sequence", "ground_truth_video", "audio"}},
        }

    # ---- Pairwise candidate list (with optional winner index) ----
    candidates = record.get("candidates") or record.get("videos") or record.get("responses")
    if candidates and len(_as_list(candidates)) >= 2:
        paths = [_resolve_video_path(record, c) for c in _as_list(candidates)]
        label_rank = record.get("label_rank")
        if label_rank is None and record.get("winner") is not None:
            winner = int(record["winner"])
            label_rank = [1, 2] if winner == 0 else [2, 1]
        item: dict[str, Any] = {
            "query": query,
            "response": paths[:2],
            "label_rank": label_rank or [1, 2],
            "task_type": _map_to_arr_task_type(task),
            "metadata": metadata,
        }
        item.update({k: v for k, v in references.items() if k in {"subject_image", "object_image", "scene_image", "depth_sequence", "ground_truth_video", "audio"}})
        if record.get("human_dims") or record.get("dimension_scores"):
            metadata["human_dims"] = _normalize_dim_scores(
                record.get("human_dims") or record.get("dimension_scores") or {},
            )
        return item

    # ---- Already canonical pairwise ----
    response = record.get("response") or record.get("video_paths")
    if isinstance(response, list) and len(response) >= 2:
        item = {
            "query": query,
            "response": [str(p) for p in response[:2]],
            "label_rank": record.get("label_rank", [1, 2]),
            "task_type": _map_to_arr_task_type(task),
            "metadata": metadata,
        }
        item.update({k: v for k, v in references.items() if k in {"subject_image", "object_image", "scene_image", "depth_sequence", "ground_truth_video", "audio"}})
        if record.get("human_dims") or record.get("dimension_scores"):
            metadata["human_dims"] = _normalize_dim_scores(
                record.get("human_dims") or record.get("dimension_scores") or {},
            )
        return item

    # ---- Pointwise multi-dimension expansion ----
    dim_scores = (
        record.get("human_dims")
        or record.get("dimension_scores")
        or record.get("scores")
    )
    video_path = _first_non_empty(
        record.get("video"),
        record.get("video_path"),
        record.get("response"),
    )
    if dim_scores and video_path:
        if isinstance(video_path, list):
            video_path = video_path[0]
        normalized = _normalize_dim_scores(dim_scores)
        cot = record.get("cot") or record.get("rationale") or record.get("chain_of_thought")
        expanded: list[dict[str, Any]] = []
        for dim_name, score in normalized.items():
            dim_meta = {
                **metadata,
                "evalverse_dimension": dim_name,
                "evalverse_aspect": aspect_of(dim_name) or "unknown",
            }
            if cot:
                dim_meta["cot_rationale"] = cot if isinstance(cot, str) else cot.get(dim_name, "")
            sample = {
                "query": query,
                "response": str(video_path),
                "label_score": int(round(float(score))),
                "dimension": dim_name,
                "task_type": _map_to_arr_task_type(task),
                "metadata": dim_meta,
            }
            sample.update({k: v for k, v in references.items() if k in {"subject_image", "object_image", "scene_image", "depth_sequence", "ground_truth_video", "audio"}})
            expanded.append(sample)
        return expanded

    # ---- Single-video pointwise fallback ----
    if video_path and record.get("label_score") is not None:
        sample = {
            "query": query,
            "response": str(video_path),
            "label_score": int(round(float(record["label_score"]))),
            "dimension": record.get("dimension"),
            "task_type": _map_to_arr_task_type(task),
            "metadata": metadata,
        }
        sample.update({k: v for k, v in references.items() if k in {"subject_image", "object_image", "scene_image", "depth_sequence", "ground_truth_video", "audio"}})
        return sample

    raise ValueError(f"Unsupported EvalVerse record shape: {record}")


def _map_to_arr_task_type(evalverse_task: str) -> str:
    """Map EvalVerse task identifiers to ARR media task_type used by Grader."""
    if evalverse_task == EVALVERSE_TASK_T2V:
        return "t2v"
    if evalverse_task == EVALVERSE_TASK_MULTISHOT:
        return "t2v_multishot"
    if evalverse_task == EVALVERSE_TASK_WITH_SOUND:
        return "t2v_sound"
    if evalverse_task in {
        EVALVERSE_TASK_R2V,
        EVALVERSE_TASK_SUBJECT_TO_VIDEO,
        EVALVERSE_TASK_OBJECT_TO_VIDEO,
        EVALVERSE_TASK_SCENE_TO_VIDEO,
        EVALVERSE_TASK_CAMERA_TO_VIDEO,
    }:
        return "r2v"
    return "t2v"


# ============================================================
# IO
# ============================================================


def load_evalverse_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load EvalVerse JSON/JSONL and convert to ARR samples."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        raw = loaded if isinstance(loaded, list) else [loaded]

    samples: list[dict[str, Any]] = []
    for record in raw:
        converted = convert_evalverse_record(record)
        if isinstance(converted, list):
            samples.extend(converted)
        else:
            samples.append(converted)
    return samples


def save_arr_dataset(samples: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Save canonical ARR samples as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(samples)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def filter_pairwise(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in samples if isinstance(s.get("response"), list) and s.get("label_rank")]


def filter_pointwise_by_dimension(
    samples: Iterable[Mapping[str, Any]],
    dimension: str | None = None,
    aspect: str | None = None,
) -> list[dict[str, Any]]:
    """Select pointwise samples by canonical dimension or parent aspect."""
    dim = _normalize_dimension(dimension) if dimension else None
    target_aspect = _normalize_dimension(aspect) if aspect else None
    out: list[dict[str, Any]] = []
    for sample in samples:
        if "label_score" not in sample:
            continue
        sample_dim = sample.get("dimension")
        if not sample_dim:
            continue
        canon = _normalize_dimension(sample_dim)
        if dim and canon != dim:
            continue
        if target_aspect and aspect_of(canon) != target_aspect:
            continue
        out.append(dict(sample))
    return out


def group_by_aspect(samples: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {asp: [] for asp in EVALVERSE_ASPECTS}
    grouped["unknown"] = []
    for sample in samples:
        dim = sample.get("dimension")
        asp = aspect_of(dim) if dim else None
        grouped.setdefault(asp or "unknown", []).append(dict(sample))
    return grouped


__all__ = [
    "EVALVERSE_ASPECTS",
    "EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS",
    "EVALVERSE_MAIN_DIMENSIONS",
    "EVALVERSE_MAIN_TO_SUB_DIMENSIONS",
    "EVALVERSE_SUB_DIMENSIONS",
    "EVALVERSE_DIMENSIONS",
    "EVALVERSE_TASK_T2V",
    "EVALVERSE_TASK_R2V",
    "EVALVERSE_TASK_MULTISHOT",
    "EVALVERSE_TASK_WITH_SOUND",
    "EVALVERSE_TASK_SUBJECT_TO_VIDEO",
    "EVALVERSE_TASK_OBJECT_TO_VIDEO",
    "EVALVERSE_TASK_SCENE_TO_VIDEO",
    "EVALVERSE_TASK_CAMERA_TO_VIDEO",
    "aspect_of",
    "convert_evalverse_record",
    "filter_pairwise",
    "filter_pointwise_by_dimension",
    "group_by_aspect",
    "load_evalverse_dataset",
    "normalize_evalverse_task",
    "save_arr_dataset",
]
