"""rm_training.stage1_teacher.task_descriptions
=============================================
Load per-dimension task descriptions + EvalVerse metadata from the
``rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml`` files. These are the
canonical "this is what evaluating dim X means" prompts that Stage ① teacher
saw when generating gold rubrics; Stage ② student must see the same
conceptual scaffolding so its emitted rubric lives in the same
sub-aspect taxonomy.

Why a dedicated module
----------------------
The yaml files carry a long ``task_description`` that mixes:
  • the *dimension definition* (what "Action" means here, sub-aspects);
  • Gemini-specific boilerplate (Pointwise calibration B3, HARD
    constraints, etc.).

For Stage ② we want only the dimension definition — the calibration /
anchor knobs apply to a different inference path (single-video pointwise
score) and would burn 7B's context budget without helping rubric
generation.

Public API
----------
- ``load_task_specs(repo_root)``
    → ``dict[dim, TaskSpec]`` for all 18 dims under config/.
- ``TaskSpec`` (dataclass)
    .dimension              str  — the dim slug, e.g. "action"
    .evalverse_aspect       str  — "Acting"
    .evalverse_main         str  — "Action"
    .task_description_short str  — first 2 paragraphs (definition + sub-aspects)
    .sub_aspects            list[str] — extracted bullets
    .task_description_full  str  — the raw yaml field, in case Stage ②
                                    wants to re-derive a different slice.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TaskSpec", "load_task_specs", "DEFAULT_CONFIG_GLOB"]


DEFAULT_CONFIG_GLOB = "rubric_pipeline/config/v_arr_mvp_*_t2v.yaml"

# Anything from ``## Pointwise scoring calibration`` (the Gemini-only
# scoring rubric) onward is dropped from the "short" version. The cut
# is unambiguous because every yaml uses this exact heading.
_BOILERPLATE_HEADERS = (
    "## Pointwise scoring calibration",
    "## Rubric requirements",
)


@dataclass
class TaskSpec:
    dimension:               str
    evalverse_aspect:        str
    evalverse_main:          str
    task_description_short:  str
    sub_aspects:             list[str] = field(default_factory=list)
    task_description_full:   str = ""

    def short(self) -> str:
        """Convenience: dimension definition only (≈ 1KB)."""
        return self.task_description_short


# ---------------------------------------------------------------------------
# Sub-aspects extractor.
#
# The yamls all use the same exact phrasing:
#
#     Sub-aspects covered by this dimension (each rubric should ground itself
#     in one or more of these):
#       - Acting-Action-Action-Emotion_Synergy
#       - Acting-Action-Action_Tension
#       - Acting-Action-Interaction_Rationality
#
#     When evaluating, the user instruction prompt IS available; ...
#
# We capture ONLY the bullet block immediately under the header, stopping
# at the first blank line. (A previous regex grabbed every bullet in the
# whole task_description, which leaked Rubric Requirements + 1-5
# calibration lines into sub_aspects.)
# ---------------------------------------------------------------------------
# Header line spans 2 lines in the yaml so we must allow a newline
# inside it. The bullet block then starts immediately after the colon,
# with one bullet per line until the first blank line.
_SUB_ASPECT_BLOCK_RE = re.compile(
    r"Sub-aspects covered by this dimension[^:]*:\s*\n"
    r"(?P<block>(?:[ \t]*-[ \t]+\S[^\n]*\n)+)",
)
_BULLET_RE = re.compile(r"^[ \t]*-[ \t]+(\S.*?)\s*$", re.MULTILINE)


def _extract_sub_aspects(task_description: str) -> list[str]:
    m = _SUB_ASPECT_BLOCK_RE.search(task_description)
    if not m:
        return []
    return _BULLET_RE.findall(m.group("block"))


# ---------------------------------------------------------------------------
# Trim the long Gemini boilerplate, keep the dimension-defining head.
# ---------------------------------------------------------------------------
def _trim_to_short(task_description: str) -> str:
    if not task_description:
        return ""
    cut_at = len(task_description)
    for header in _BOILERPLATE_HEADERS:
        idx = task_description.find(header)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return task_description[:cut_at].rstrip() + "\n"


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------
def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as e:                                      # pragma: no cover
        raise RuntimeError(
            "PyYAML required for task spec loading — pip install pyyaml"
        ) from e
    return yaml


def load_task_specs(
    repo_root: Path | str,
    *,
    glob_pattern: str = DEFAULT_CONFIG_GLOB,
    add_visual_quality_fallback: bool = True,
) -> dict[str, TaskSpec]:
    """Scan ``<repo_root>/<glob_pattern>``; return ``{dim: TaskSpec}``.

    Missing dims (no yaml) are simply absent from the dict — caller
    decides whether to fall back to a generic prompt or skip the dim.

    ``visual_quality`` has no dedicated yaml in the current repo; it
    inherits the generic ``v_arr_mvp_t2v.yaml`` task description. When
    ``add_visual_quality_fallback`` is True we wire it up so callers
    enumerating all 18 EvalVerse dims see a TaskSpec for it.
    """
    yaml = _import_yaml()
    root = Path(repo_root)
    paths = sorted(root.glob(glob_pattern))
    if not paths:
        print(f"[task_specs] no yaml matched {root}/{glob_pattern}", file=sys.stderr)
        return {}

    out: dict[str, TaskSpec] = {}
    for fp in paths:
        try:
            cfg = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[task_specs] {fp.name} unreadable: {e}", file=sys.stderr)
            continue
        dim = cfg.get("dimension")
        td  = cfg.get("task_description") or ""
        if not dim or not td:
            continue
        spec = TaskSpec(
            dimension              = str(dim),
            evalverse_aspect       = str(cfg.get("evalverse_aspect", "") or ""),
            evalverse_main         = str(cfg.get("evalverse_main", "") or ""),
            task_description_short = _trim_to_short(td),
            sub_aspects            = _extract_sub_aspects(td),
            task_description_full  = td,
        )
        out[spec.dimension] = spec

    if add_visual_quality_fallback and "visual_quality" not in out:
        out["visual_quality"] = _visual_quality_fallback(root)

    return out


def _visual_quality_fallback(root: Path) -> TaskSpec:
    """Synthesize a TaskSpec for visual_quality from the base yaml.

    The base ``v_arr_mvp_t2v.yaml`` is the generic spatial+temporal judge
    that Stage ① actually used to generate ``v_arr_mvp_vq_v2_prompt_pair.txt``.
    sub_aspects stays empty because the generic prompt deliberately avoids
    a fixed taxonomy.
    """
    yaml = _import_yaml()
    fp   = root / "rubric_pipeline" / "config" / "v_arr_mvp_t2v.yaml"
    td   = ""
    try:
        cfg = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        td  = cfg.get("task_description") or ""
    except Exception:
        pass
    if not td:
        # Last-ditch: a hand-written paragraph so the dim isn't unusable.
        td = (
            "This rubric judge evaluates the **visual_quality** dimension: "
            "per-frame visual fidelity (sharpness, structural integrity), "
            "temporal stability of objects/characters/backgrounds, and the "
            "absence of rendering artifacts (warping, melting, watermarks).\n"
        )
    return TaskSpec(
        dimension              = "visual_quality",
        evalverse_aspect       = "Aesthetics",
        evalverse_main         = "Visual_Quality",
        task_description_short = _trim_to_short(td),
        sub_aspects            = [],            # generic prompt has no fixed taxonomy
        task_description_full  = td,
    )
