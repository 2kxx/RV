"""RewardVerse MVP — EvalVerse 18 second-level dimension registry.

This module is the single source of truth for "which sub-dimension keys
belong to which second-level dimension" plus a couple of per-dimension
behavioural flags that the build script and the rubric-generation pipeline
need to consult.

Structure of the source data (per record in
``merged_evaluate_results_with_path_part*.jsonl``):

    record["evaluate_data"][i]["evaluate_result"] = {
        "Acting-Action-Action-Emotion_Synergy": {"average_score": 0.88, ...},
        "Acting-Action-Action_Tension":          {"average_score": ..., ...},
        ...
    }

Each key is "Aspect-Main-Sub" (the Multi/Shot_Cutting bucket has compound
sub keys "Logic-*" and "Rhythm-*" which we treat as two separate
second-level dimensions in this registry).

Per-dimension flags:
  * ``expose_prompt``: whether to feed the user instruction prompt to Gemini
    when generating / applying rubrics for this dimension. The four
    "intrinsic" dimensions (Logic / Progression / Visual_Quality /
    Consistency) are scored without seeing the prompt, per the user's
    request (2026-05-21).
  * ``needs_audio``: whether the dimension requires the audio track. Only
    Vocal and Soundscape need this; venus llmproxy + Gemini 3.x consume
    the audio track natively when video is sent as a multimodal_url block,
    so no separate ffmpeg extraction is required — we just point this out
    in the per-dim yaml task_description.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DimensionSpec:
    """One second-level EvalVerse dimension."""

    name: str                              # snake_case slug used everywhere
    aspect: str                            # EvalVerse top-level aspect
    main: str                              # EvalVerse second-level main name
    sub_keys: tuple[str, ...]              # full keys to look up in evaluate_result
    description_zh: str
    description_en: str
    expose_prompt: bool = True
    needs_audio: bool = False
    sub_key_filter: str | None = field(default=None)  # only for Logic/Rhythm split


# -----------------------------------------------------------------------------
# 18 dimension specs (17 new + the existing Visual_Quality kept for reference).
# -----------------------------------------------------------------------------
DIMENSIONS: tuple[DimensionSpec, ...] = (
    # --- Visual_Concept_Design ------------------------------------------------
    DimensionSpec(
        name="character",
        aspect="Visual_Concept_Design",
        main="Character",
        sub_keys=(
            "Visual_Concept_Design-Character-Costume_Rationality",
            "Visual_Concept_Design-Character-Identifiability",
        ),
        description_zh="角色设定：评估角色外观是否清晰、可识别，并符合设定。",
        description_en=(
            "Character design: assess whether each character's appearance "
            "is clear, identifiable, and consistent with the setting."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="scene",
        aspect="Visual_Concept_Design",
        main="Scene",
        sub_keys=(
            "Visual_Concept_Design-Scene-Environment_Plausibility",
            "Visual_Concept_Design-Scene-Genre_Distinctiveness",
        ),
        description_zh=(
            "场景设定：评估世界及场景是否成立且具有吸引力；评估是否形成"
            "明确、可识别、有审美价值的整体视觉风格。"
        ),
        description_en=(
            "Scene design: assess whether the world and scenes are "
            "coherent and compelling, and whether the work establishes a "
            "clear, identifiable, aesthetically valuable overall visual style."
        ),
        expose_prompt=True,
    ),

    # --- Acting ---------------------------------------------------------------
    DimensionSpec(
        name="consistency",
        aspect="Acting",
        main="Consistency",
        sub_keys=(
            "Acting-Consistency-Attribute",
            "Acting-Consistency-Face_Identity",
        ),
        description_zh="一致性：评估角色属性与身份在跨帧、跨镜头中的稳定性。",
        description_en=(
            "Consistency: assess whether character attributes and "
            "identity remain stable across frames and across shots."
        ),
        expose_prompt=False,  # intrinsic, judged purely from video
    ),
    DimensionSpec(
        name="action",
        aspect="Acting",
        main="Action",
        sub_keys=(
            "Acting-Action-Action-Emotion_Synergy",
            "Acting-Action-Action_Tension",
            "Acting-Action-Interaction_Rationality",
        ),
        description_zh=(
            "动作与行为表现力：评估角色动作是否自然、符合物理规律并契合"
            "角色身份。"
        ),
        description_en=(
            "Action / behavioural expressiveness: assess whether character "
            "motion is natural, physically plausible, and faithful to the "
            "character's identity and emotional state."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="expression",
        aspect="Acting",
        main="Expression",
        sub_keys=(
            "Acting-Expression-Accuracy",
            "Acting-Expression-Continuity",
            "Acting-Expression-Expression_Diversity",
            "Acting-Expression-Facial_Tension",
        ),
        description_zh=(
            "表情状态表达：评估角色面部表情是否准确、生动且传达预期情绪。"
        ),
        description_en=(
            "Facial expression: assess whether the character's facial "
            "expressions are accurate, vivid, and convey the intended emotion."
        ),
        expose_prompt=True,
    ),

    # --- Cinematography -------------------------------------------------------
    DimensionSpec(
        name="composition",
        aspect="Cinematography",
        main="Composition",
        sub_keys=(
            "Cinematography-Composition-Shot-Size_Rationality",
            "Cinematography-Composition-Spatial_Layering",
            "Cinematography-Composition-Subject_Prominence",
        ),
        description_zh=(
            "镜头设计与构图：评估单个镜头是否清晰、稳定且具备明确视觉表达意图。"
        ),
        description_en=(
            "Shot design and composition: assess whether each shot is "
            "clear, stable, and exhibits a deliberate visual intent."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="lens",
        aspect="Cinematography",
        main="Lens",
        sub_keys=(
            "Cinematography-Lens-Depth_of_Field",
            "Cinematography-Lens-Exposure",
            "Cinematography-Lens-Focal_Length",
            "Cinematography-Lens-Focus",
        ),
        description_zh=(
            "镜头参数与摄影设置合理性：评估画面中所呈现的“镜头参数选择”"
            "是否清晰、合理、稳定。"
        ),
        description_en=(
            "Lens parameters / cinematographic settings: assess whether "
            "the chosen depth-of-field, exposure, focal length and focus "
            "are clear, reasonable, and stable."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="pacing",
        aspect="Cinematography",
        main="Pacing",
        sub_keys=(
            "Cinematography-Pacing-Movement_Rationality",
        ),
        description_zh="镜头运动与调度：评估镜头运动方式是否合理。",
        description_en=(
            "Camera movement / blocking: assess whether the choreography "
            "of camera motion is reasonable."
        ),
        expose_prompt=True,
    ),

    # --- Aesthetics -----------------------------------------------------------
    DimensionSpec(
        name="chromaticity",
        aspect="Aesthetics",
        main="Chromaticity",
        sub_keys=(
            "Aesthetics-Chromaticity-Emotive_Power",
            "Aesthetics-Chromaticity-Harmony",
        ),
        description_zh="色彩：评估画面色彩的情绪表达力与整体协调性。",
        description_en=(
            "Chromaticity: assess the emotive power and overall harmony "
            "of the colour design."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="materiality",
        aspect="Aesthetics",
        main="Materiality",
        sub_keys=(
            "Aesthetics-Materiality-Material_Identifiability",
            "Aesthetics-Materiality-Stylistic_Consistency",
        ),
        description_zh=(
            "材质、质感与细节表现：评估画面质感是否美观、自洽、可信、精致。"
        ),
        description_en=(
            "Materiality / texture / detail: assess whether material "
            "rendering is aesthetic, self-consistent, believable and refined."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="lighting",
        aspect="Aesthetics",
        main="Lighting",
        sub_keys=(
            "Aesthetics-Lighting-Lighting_Logic",
            "Aesthetics-Lighting-Volumetric_Sculpting",
        ),
        description_zh=(
            "光影设计与照明逻辑：评估光影在美学与影视逻辑层面的表现。"
        ),
        description_en=(
            "Lighting design / lighting logic: assess the aesthetic and "
            "cinematographic correctness of the lighting."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="visual_quality",
        aspect="Aesthetics",
        main="Visual_Quality",
        sub_keys=(
            "Aesthetics-Visual_Quality-Physics",
            "Aesthetics-Visual_Quality-Rendering_Quality",
            "Aesthetics-Visual_Quality-Temporal_Consistency",
        ),
        description_zh=(
            "画面品质：评估物理合理性、渲染质量、时序一致性等基础视频质量。"
        ),
        description_en=(
            "Visual quality: assess physics plausibility, rendering "
            "quality, and temporal consistency as foundational video quality."
        ),
        expose_prompt=False,  # intrinsic
    ),

    # --- Affectivity ----------------------------------------------------------
    DimensionSpec(
        name="grounding",
        aspect="Affectivity",
        main="Grounding",
        sub_keys=(
            "Affectivity-Grounding-Tonal_Identifiability",
            "Affectivity-Grounding-Visual-Emotional_Synergy",
        ),
        description_zh=(
            "情绪与氛围定位：评估作品是否建立了清晰、稳定、可识别的整体"
            "情绪方向与氛围设定。"
        ),
        description_en=(
            "Emotional grounding / atmosphere: assess whether the work "
            "establishes a clear, stable, identifiable emotional direction "
            "and atmosphere."
        ),
        expose_prompt=True,
    ),
    DimensionSpec(
        name="progression",
        aspect="Affectivity",
        main="Progression",
        sub_keys=(
            "Affectivity-Progression-Intensity_Layering",
            "Affectivity-Progression-Transition_Continuity",
        ),
        description_zh=(
            "情绪推进与节奏控制：评估情绪在时间维度上的建立、变化与收束"
            "是否自然、有控制。"
        ),
        description_en=(
            "Emotional progression / pacing of emotion: assess whether "
            "emotional build-up, change, and resolution unfold naturally "
            "and with deliberate control over time."
        ),
        expose_prompt=False,  # intrinsic
    ),

    # --- Multi > Shot_Cutting (split into two dimensions) --------------------
    DimensionSpec(
        name="logic",
        aspect="Multi",
        main="Shot_Cutting",
        # The 4 Logic-* sub keys, fully qualified.
        sub_keys=(
            "Multi-Shot_Cutting-Logic-Multi-shot_Rationality",
            "Multi-Shot_Cutting-Logic-Narrative_Continuity",
            "Multi-Shot_Cutting-Logic-Scene_Consistency",
            "Multi-Shot_Cutting-Logic-Spatial_Continuity",
        ),
        sub_key_filter="Logic-",
        description_zh=(
            "镜头衔接与剪辑逻辑：评估镜头之间在时间、空间与动作上的衔接"
            "是否连贯，剪辑是否有效帮助观众理解内容。"
        ),
        description_en=(
            "Shot transitions / editing logic: assess whether shots "
            "connect coherently in time, space, and action, and whether "
            "the cut effectively conveys the narrative."
        ),
        expose_prompt=False,  # intrinsic
    ),
    DimensionSpec(
        name="rhythm",
        aspect="Multi",
        main="Shot_Cutting",
        sub_keys=(
            "Multi-Shot_Cutting-Rhythm-Rhythmic_Layering",
            "Multi-Shot_Cutting-Rhythm-Shot_Duration_Rationality",
        ),
        sub_key_filter="Rhythm-",
        description_zh=(
            "节奏与时序控制：评估镜头时长与剪辑节奏的整体控制是否合理，"
            "是否通过节奏变化有效引导情绪并清晰组织时间推进。"
        ),
        description_en=(
            "Rhythm / temporal control: assess whether shot durations and "
            "cutting cadence are well controlled, and whether rhythm "
            "shifts effectively guide emotion and organise the timeline."
        ),
        expose_prompt=True,
    ),

    # --- Sound_Design (audio-required) ---------------------------------------
    DimensionSpec(
        name="vocal",
        aspect="Sound_Design",
        main="Vocal",
        sub_keys=(
            "Sound_Design-Vocal-Acoustic_Quality",
            "Sound_Design-Vocal-Lip-Sync",
        ),
        description_zh=(
            "人声：评估人声音质与口型同步是否自然真实。"
        ),
        description_en=(
            "Vocal: assess naturalness of vocal acoustic quality and "
            "lip-sync alignment with on-screen mouth motion."
        ),
        expose_prompt=True,
        needs_audio=True,
    ),
    DimensionSpec(
        name="soundscape",
        aspect="Sound_Design",
        main="Soundscape",
        sub_keys=(
            "Sound_Design-Soundscape-Ambient_Sound_Fidelity",
            "Sound_Design-Soundscape-Musical_Score_Alignment",
            "Sound_Design-Soundscape-Narrative_Sound_Design",
        ),
        description_zh=(
            "声音设计与画面关系：评估声音元素（环境音、音效、音乐）是否"
            "与画面形成有机配合，在同步、层次与叙事功能上共同构建视听体验。"
        ),
        description_en=(
            "Sound design / audio-visual relationship: assess whether "
            "ambient sound, sound effects and musical score work "
            "organically with the visuals in synchrony, layering and "
            "narrative function."
        ),
        expose_prompt=True,
        needs_audio=True,
    ),
)


# -----------------------------------------------------------------------------
# Convenience lookups.
# -----------------------------------------------------------------------------
DIMENSIONS_BY_NAME: dict[str, DimensionSpec] = {d.name: d for d in DIMENSIONS}


def get_dimension(name: str) -> DimensionSpec:
    """Look up a dimension by its snake_case slug. Raises KeyError if missing."""
    name = name.lower().strip()
    if name not in DIMENSIONS_BY_NAME:
        raise KeyError(
            f"unknown dimension {name!r}; valid names: "
            f"{sorted(DIMENSIONS_BY_NAME)}"
        )
    return DIMENSIONS_BY_NAME[name]


def list_dimension_names() -> list[str]:
    """All 18 second-level dimension slugs, in EvalVerse's logical order."""
    return [d.name for d in DIMENSIONS]


# ---------------------------------------------------------------------------
# Test-set path resolution.
# ---------------------------------------------------------------------------
#
# The 18 dimensions split into two test-set conventions:
#
#   * visual_quality: reuses the existing
#       data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json
#     which was hand-curated by build_visual_quality_mvp_dataset.py to span
#     the full 1.04-4.77 label range with 5-band stratification.
#
#   * everything else: uses
#       data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json
#     emitted by ``build_dim_seed_v2.py --mode test_pointwise``.
#
# Centralising this rule here avoids hard-coding the visual_quality
# special-case in every downstream evaluation driver.

def get_test_set_path(name: str, project_root: "str | Path | None" = None) -> "Path":
    """Return the canonical pointwise test-set path for a dimension.

    Parameters
    ----------
    name
        Dimension slug (e.g. ``"character"``, ``"visual_quality"``).
    project_root
        Repo root; defaults to the parent of ``scripts/v_arr_mvp/``. Pass an
        absolute Path when calling from a script that ``chdir``-ed elsewhere.
    """
    from pathlib import Path as _Path
    if project_root is None:
        project_root = _Path(__file__).resolve().parents[2]
    else:
        project_root = _Path(project_root)
    n = name.strip().lower()
    if n not in DIMENSIONS_BY_NAME:
        raise KeyError(
            f"unknown dimension {name!r}; valid names: {sorted(DIMENSIONS_BY_NAME)}"
        )
    if n == "visual_quality":
        return project_root / "data" / "v_arr_mvp" / "visual_quality_mvp" / "shared_test_pointwise.json"
    return project_root / "data" / "v_arr_mvp" / f"{n}_mvp" / f"test_pointwise_{n}.json"


if __name__ == "__main__":  # smoke test
    import sys
    print(f"Total dimensions: {len(DIMENSIONS)}")
    for d in DIMENSIONS:
        flags = []
        if not d.expose_prompt:
            flags.append("no-prompt")
        if d.needs_audio:
            flags.append("needs-audio")
        flag_str = f"  [{','.join(flags)}]" if flags else ""
        print(f"  {d.name:<14s} ({d.aspect}>{d.main}, {len(d.sub_keys)} sub)"
              f"{flag_str}")
    sys.exit(0)
