"""rm_training.stage2_grpo.data_collator
=====================================
Light-weight collator for the (q, y⁺, y⁻) Stage ② training rows.

We do NOT inherit from any HF Data class because:
  • the GRPO loop drives sampling itself (G rollouts per row);
  • the heavy lifting (video preprocessing) lives inside the student
    wrapper, not in the collator.

What this module does
---------------------
1. ``load_jsonl(path) -> list[dict]``        — read Stage ① jsonl
2. ``rebase_video_paths``                    — make jsonl paths portable
3. ``GoldRubricDataset``                     — torch-style ``__getitem__``
4. ``collate_pair_for_grpo(rows) -> dict``   — minimal field repack

Path portability
----------------
Stage ①'s ``build_gold_rubric_dataset_from_txt.py`` writes ABSOLUTE
paths into ``y_pos`` / ``y_neg`` (e.g. ``/group/40063/.../cache_videos/<id>.mp4``).
When the jsonl is moved to a different machine — e.g. from the original
GPU host (``/group/40063/jacuzzitang/projects/RewardVerse``) to the
8×H100 host (``/efs/jacuzzitang/projects/RewardVerse``) — those paths
break. ``rebase_video_paths`` rewrites every ``y_pos``/``y_neg`` so the
basename is resolved relative to the current ``RewardVerse`` repo root,
without re-running stage1.

Intentionally minimal so train_grpo.py can plug new fields (caption,
domain id, etc.) without touching this file.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "load_jsonl",
    "rebase_video_paths",
    "attach_score_targets",
    "GoldRubricDataset",
    "collate_pair_for_grpo",
]


# Match anything that LOOKS like a previous-host RewardVerse data path.
# Captures everything from "data/v_arr_mvp/..." onward so we can re-attach
# it to the current ROOT.
_DATA_PREFIX_RE = re.compile(r"(/.*/(?:RewardVerse|rewardverse))/(data/v_arr_mvp/)")


def _rebase_one(p: str, root: Path) -> str:
    """Rewrite ``p`` so that everything from ``data/v_arr_mvp/`` onward
    is resolved under ``root``.

    Pass-through cases:
      * empty / non-string;
      * URL (http / https);
      * already a valid local file under any path → leave as-is;
      * doesn't contain ``data/v_arr_mvp/`` → unchanged (caller's
        responsibility).
    """
    if not isinstance(p, str) or not p:
        return p
    if p.startswith(("http://", "https://")):
        return p
    if Path(p).is_file():
        return p

    # Strategy 1: regex match for /<anywhere>/RewardVerse/data/v_arr_mvp/...
    m = _DATA_PREFIX_RE.match(p)
    if m:
        # Take the suffix after ``data/v_arr_mvp/`` and reattach to ROOT.
        suffix = p[m.end(1):]   # e.g. "/data/v_arr_mvp/cache_videos/foo.mp4"
        candidate = root / suffix.lstrip("/")
        if candidate.is_file():
            return str(candidate)

    # Strategy 2: scan for the substring ``data/v_arr_mvp/`` and reattach.
    idx = p.find("data/v_arr_mvp/")
    if idx >= 0:
        candidate = root / p[idx:]
        if candidate.is_file():
            return str(candidate)

    # Strategy 3: last-ditch — try just the basename inside cache dirs.
    base = Path(p).name
    for cache_dir in (
        root / "data" / "v_arr_mvp" / "cache_videos",
        root / "runs" / "baselines" / "_video_cache",
    ):
        candidate = cache_dir / base
        if candidate.is_file():
            return str(candidate)

    # Couldn't rebase — return original; callers may then surface a
    # clearer "video not found" error than torchvision's stack trace.
    return p


def rebase_video_paths(rows: list[dict], root: Path | str) -> list[dict]:
    """Re-resolve ``y_pos`` / ``y_neg`` in every row against ``root``.

    Mutates rows in-place AND returns them, so callers can chain.
    Reports how many rows were rewritten / unresolved on stderr.
    """
    root = Path(root)
    rebased = 0
    unresolved = 0
    for r in rows:
        for k in ("y_pos", "y_neg"):
            old = r.get(k)
            if not old:
                continue
            new = _rebase_one(old, root)
            if new != old:
                r[k] = new
                rebased += 1
            elif not (Path(old).is_file() or old.startswith(("http://", "https://"))):
                unresolved += 1
    if rebased or unresolved:
        print(f"[data] rebased {rebased} video paths to {root}; "
              f"{unresolved} still unresolved.", file=sys.stderr)
    return rows


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def attach_score_targets(rows: list[dict], root: Path | str) -> list[dict]:
    """Join ``winner_score`` / ``loser_score`` from each row's
    ``source_pair_file`` (the original ``v2_prompt_pair.json``) onto the
    row as ``s_pos_target`` / ``s_neg_target`` in [0, 1].

    Why this lives here
    -------------------
    Stage ① jsonl carries only ``gt_label`` and the gold rubric — pointwise
    + score-gap supervision needs the absolute winner / loser scores. The
    pair file referenced by ``source_pair_file`` has them under
    ``metadata.winner_score`` / ``metadata.loser_score`` (range 0-1) plus
    ``response[0]`` / ``response[1]`` URLs. We match by ``pair_id`` and
    cross-check ``url_pos`` / ``url_neg`` against ``response[0/1]`` to
    handle the ``gt_label == "B"`` case where url_pos = response[1].

    Failure mode is non-fatal: rows whose source file is missing, or whose
    metadata lacks the scores, are left untouched (callers must check
    ``s_pos_target`` is not None before using it).

    Returns the same list (mutated in-place).
    """
    root = Path(root)
    pair_cache: dict[Path, dict[str, dict]] = {}
    matched, missing_file, missing_id = 0, 0, 0
    for r in rows:
        src = r.get("source_pair_file")
        if not src:
            continue
        # source_pair_file is repo-relative (e.g. data/v_arr_mvp/...).
        src_path = (root / src).resolve()
        if src_path not in pair_cache:
            if not src_path.is_file():
                pair_cache[src_path] = {}
                missing_file += 1
            else:
                with open(src_path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                pair_cache[src_path] = {
                    rec.get("id"): rec for rec in arr if isinstance(rec, dict)
                }
        idx = pair_cache[src_path].get(r.get("pair_id"))
        if idx is None:
            missing_id += 1
            continue
        meta = idx.get("metadata") or {}
        ws = meta.get("winner_score")
        ls = meta.get("loser_score")
        if ws is None or ls is None:
            continue
        # ``gt_label`` is 'A' when y_pos = response[0] and 'B' when
        # y_pos = response[1]. winner_score always pairs with the
        # *winner* method (response[0] in this dataset), so swap when
        # gt_label == 'B'. Cross-check via url_pos / url_neg if present.
        gt = (r.get("gt_label") or "").upper()
        responses = idx.get("response") or []
        url_pos = r.get("url_pos")
        if (url_pos and len(responses) >= 1
                and isinstance(responses[0], str)
                and url_pos == responses[0]):
            s_pos, s_neg = float(ws), float(ls)
        elif (url_pos and len(responses) >= 2
                and isinstance(responses[1], str)
                and url_pos == responses[1]):
            s_pos, s_neg = float(ls), float(ws)
        elif gt == "B":
            s_pos, s_neg = float(ls), float(ws)
        else:
            s_pos, s_neg = float(ws), float(ls)
        r["s_pos_target"] = s_pos
        r["s_neg_target"] = s_neg
        matched += 1
    if matched or missing_file or missing_id:
        print(
            f"[data] score-target join: matched={matched} "
            f"missing_src={missing_file} missing_pair_id={missing_id} "
            f"of {len(rows)} rows",
            file=sys.stderr,
        )
    return rows


@dataclass
class GoldRubricDataset:
    """Tiny in-memory dataset around the Stage ① jsonl.

    Each item resolves to a plain dict carrying::

        {
          "pair_id":     str,
          "dimension":   str,
          "query":       str,
          "y_pos":       str,         # video path
          "y_neg":       str,
          "gold_rubric": list[dict],  # [{path, weight, evidence_*}]
          "gt_label":    str,         # "A"/"B"/"same"
        }
    """
    rows: list[dict]

    @classmethod
    def from_jsonl(cls, path: str | Path,
                   *, rebase_root: Path | str | None = None,
                   ) -> "GoldRubricDataset":
        """Read jsonl and (optionally) rebase video paths to ``rebase_root``.

        ``rebase_root`` defaults to the project root inferred from
        ``path``'s 3rd ancestor (``data/rm_train/<file>.jsonl`` →
        repo root). Pass an explicit ROOT when the heuristic doesn't
        match (e.g. running from a vendored copy of the repo).
        """
        rows = load_jsonl(path)
        if rebase_root is None:
            # Heuristic: assume jsonl lives at <ROOT>/data/rm_train/<*.jsonl>.
            try:
                rebase_root = Path(path).resolve().parents[2]
            except IndexError:
                rebase_root = Path.cwd()
        rebase_video_paths(rows, rebase_root)
        attach_score_targets(rows, rebase_root)
        return cls(rows=rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return dict(self.rows[idx])

    def __iter__(self) -> Iterable[dict]:
        for r in self.rows:
            yield dict(r)


def collate_pair_for_grpo(rows: list[dict]) -> dict:
    """Repack a list of dataset items into batched lists keyed by field.

    The GRPO loop typically consumes one row at a time (B=1, G rollouts
    per row), so this collator's main job is just to enforce field
    presence and turn missing entries into safe empties.
    """
    keys = ["pair_id", "dimension", "query", "y_pos", "y_neg",
            "gold_rubric", "gt_label", "s_pos_target", "s_neg_target"]
    out: dict[str, list] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            default = "" if k not in ("gold_rubric", "s_pos_target", "s_neg_target") else ([] if k == "gold_rubric" else None)
            out[k].append(r.get(k, default))
    return out
