"""rm_training.stage1_teacher.build_gold_rubric_dataset_from_txt
==============================================================
Stage ① (post-teacher-run) — assemble the GRPO training jsonl.

Background
----------
The original Stage ① plan was to call ``pairwise_rubric_judge.py`` (which
itself calls Gemini) for every pair, harvest a per-pair gold rubric +
evidence, and write one jsonl row per pair. **That run is already done
offline**, but it produced *per-dimension* rubric scaffolds rather than
*per-pair* ones — they live as plain text under::

    rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt
    rubric_pipeline/rubrics/v_arr_mvp_vq_v2_prompt_pair.txt   # VQ exception

So Stage ① collapses to a pure **assembly** step: for each pair in
``data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json``, attach the parsed
rubric scaffold of its dimension and emit one jsonl row matching the
schema Stage ② already consumes (``GoldRubricDataset.from_jsonl``).

Schema written
--------------
::

    {
      "pair_id":     "varr_vq_000",
      "dimension":   "visual_quality",
      "query":       "<full prompt with [shot_0_prompt] ...>",
      "y_pos":       "<local path or URL>",
      "y_neg":       "<local path or URL>",
      "gt_label":    "A" | "B" | "same",
      "gold_rubric": [
        {"theme": "Theme: ...", "weight": 0.2, "tips": ["...", ...]},
        ... × K            # K=5 for every dim, but the parser tolerates ≠5
      ],
      "rubric_source_file":  "rubric_pipeline/rubrics/...",
      "source_pair_file":    "data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json"
    }

Notes
-----
* Pairs whose ``label_rank == [1, 1]`` (ties) are dropped — Stage ②
  reward needs a strict y⁺/y⁻ ordering.
* Local paths come from the same cache discovery as
  ``rm_training/stage0_diagnosis/build_pair_data.py`` (so we reuse the
  236 already-cached videos plus anything ``--prefetch_missing`` pulls).
* ``--require_local`` drops rows whose videos can't be resolved.

CLI
---
::

    python rm_training/stage1_teacher/build_gold_rubric_dataset_from_txt.py \\
        --pair_glob   "data/v_arr_mvp/*_mvp/v2_prompt_pair.json" \\
        --rubric_dir  rubric_pipeline/rubrics \\
        --output      data/rm_train/gold_rubrics_18d.jsonl \\
        --split_outdir data/rm_train/splits \\
        --require_local
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_training.stage0_diagnosis.build_pair_data import resolve_local
from rm_training.stage1_teacher.parse_v2_txt     import (
    DIM_TO_RUBRIC_FILENAME,
    discover_rubric_dir,
    parse_rubric_file,
)


# ---------------------------------------------------------------------------
# Helpers (mirror stage0 + the old build_gold_rubric_dataset.py).
# ---------------------------------------------------------------------------
def _dim_from_folder(folder_name: str) -> str:
    """``visual_quality_mvp`` → ``visual_quality``."""
    return folder_name[:-4] if folder_name.endswith("_mvp") else folder_name


def _label_rank_to_gt(rank) -> str:
    """``[1,2]`` → ``A``  /  ``[2,1]`` → ``B``  /  anything else → ``same``."""
    if not isinstance(rank, (list, tuple)) or len(rank) != 2:
        return "same"
    try:
        a, b = int(rank[0]), int(rank[1])
    except (TypeError, ValueError):
        return "same"
    if a < b: return "A"
    if a > b: return "B"
    return "same"


def _y_pos_neg(responses: list, label_rank) -> tuple[str, str] | None:
    """Pick (y_pos, y_neg) according to label_rank. Returns None on tie/malformed."""
    if not isinstance(responses, list) or len(responses) < 2:
        return None
    if not isinstance(label_rank, (list, tuple)) or len(label_rank) != 2:
        return None
    try:
        a, b = int(label_rank[0]), int(label_rank[1])
    except (TypeError, ValueError):
        return None
    if a == b:
        return None  # tie: drop
    if a < b:
        return responses[0], responses[1]
    return responses[1], responses[0]


# ---------------------------------------------------------------------------
# Distillation: one dim → list[row].
# ---------------------------------------------------------------------------
def distil_one_dim(
    pair_file:      Path,
    dim:            str,
    rubric_dir:     Path,
    *,
    baseline_cache: Path,
    prefetch_cache: Path,
    require_local:  bool,
) -> list[dict]:
    """Yield one jsonl row per non-tie pair in ``pair_file``."""
    rubric_filename = DIM_TO_RUBRIC_FILENAME.get(dim, f"v_arr_mvp_{dim}_v2.txt")
    rubric_path = rubric_dir / rubric_filename
    gold_rubric = parse_rubric_file(rubric_path)
    if not gold_rubric:
        print(f"[stage1] {dim}: no rubric parsed from {rubric_path} — skipped",
              file=sys.stderr)
        return []

    try:
        recs = json.loads(pair_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[stage1] {pair_file} unreadable: {e}", file=sys.stderr)
        return []

    rows: list[dict] = []
    for r in recs:
        pid       = r.get("id")
        query     = r.get("query") or r.get("prompt") or ""
        responses = r.get("response") or []
        label     = r.get("label_rank")
        if not pid:
            continue

        ypos_yneg = _y_pos_neg(responses, label)
        if ypos_yneg is None:
            continue
        y_pos_url, y_neg_url = ypos_yneg

        # Cache lookup (URL → local mp4) — mirror stage0's logic.
        lp_pos = resolve_local(y_pos_url,
                               baseline_cache=baseline_cache,
                               prefetch_cache=prefetch_cache)
        lp_neg = resolve_local(y_neg_url,
                               baseline_cache=baseline_cache,
                               prefetch_cache=prefetch_cache)
        if require_local and (lp_pos is None or lp_neg is None):
            continue

        rows.append({
            "pair_id":            pid,
            "dimension":          dim,
            "query":              query,
            "y_pos":              lp_pos or y_pos_url,
            "y_neg":              lp_neg or y_neg_url,
            "url_pos":            y_pos_url,
            "url_neg":            y_neg_url,
            "gt_label":           _label_rank_to_gt(label),
            "gold_rubric":        gold_rubric,
            "rubric_source_file": str(rubric_path.relative_to(ROOT))
                                  if rubric_path.is_relative_to(ROOT) else str(rubric_path),
            "source_pair_file":   str(pair_file.relative_to(ROOT))
                                  if pair_file.is_relative_to(ROOT) else str(pair_file),
        })
    return rows


# ---------------------------------------------------------------------------
# Stratified 90/10 split.
# ---------------------------------------------------------------------------
def split_train_val(rows: list[dict], val_frac: float, seed: int) -> tuple[list, list]:
    """Stratify by ``dimension`` so val coverage is balanced."""
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for dim, lst in by_dim.items():
        rng.shuffle(lst)
        n_val = max(1, int(round(len(lst) * val_frac))) if len(lst) >= 5 else 0
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def build(args: argparse.Namespace) -> None:
    rubric_dir = (
        Path(args.rubric_dir)
        if Path(args.rubric_dir).is_absolute()
        else (ROOT / args.rubric_dir).resolve()
    )
    if not rubric_dir.is_dir():
        # Default fall-back: derive from repo root.
        rubric_dir = discover_rubric_dir(ROOT)
    print(f"[stage1] rubric_dir = {rubric_dir}", file=sys.stderr)

    pair_files = sorted((ROOT).glob(args.pair_glob))
    if not pair_files:
        pair_files = sorted(Path(".").glob(args.pair_glob))
    if not pair_files:
        raise SystemExit(f"[stage1] no pair files matched {args.pair_glob}")
    print(f"[stage1] found {len(pair_files)} pair files", file=sys.stderr)

    baseline_cache = (ROOT / args.baseline_cache_dir).resolve()
    prefetch_cache = (ROOT / args.prefetch_cache_dir).resolve()
    output_jsonl   = (ROOT / args.output).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    by_dim_count: Counter[str] = Counter()
    missing_rubric: list[str] = []
    excluded_dims = set(args.exclude_dims or [])
    skipped_excluded: dict[str, int] = {}
    for pf in pair_files:
        dim = _dim_from_folder(pf.parent.name)
        if dim in excluded_dims:
            # Count rows we would have produced, for sanity in the summary.
            try:
                n_in = len(json.loads(pf.read_text(encoding="utf-8")))
            except Exception:
                n_in = 0
            skipped_excluded[dim] = n_in
            continue
        dim_rows = distil_one_dim(
            pf, dim, rubric_dir,
            baseline_cache=baseline_cache,
            prefetch_cache=prefetch_cache,
            require_local=args.require_local,
        )
        if not dim_rows:
            missing_rubric.append(dim)
            continue
        rows.extend(dim_rows)
        by_dim_count[dim] += len(dim_rows)

    if not rows:
        raise SystemExit("[stage1] no rows produced — check rubric_dir / pair_glob")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    print(f"[stage1] wrote {len(rows)} rows -> {output_jsonl}", file=sys.stderr)

    if args.val_frac > 0:
        split_outdir = (ROOT / args.split_outdir).resolve()
        split_outdir.mkdir(parents=True, exist_ok=True)
        train, val = split_train_val(rows, args.val_frac, args.seed)
        for name, lst in (("train", train), ("val", val)):
            fp = split_outdir / f"{name}.jsonl"
            with fp.open("w", encoding="utf-8") as f:
                for r in lst:
                    f.write(json.dumps(r, ensure_ascii=False))
                    f.write("\n")
            print(f"[stage1] split {name}: {len(lst):4d} -> {fp}", file=sys.stderr)

    print("[stage1] per-dim counts:", file=sys.stderr)
    for d in sorted(by_dim_count):
        print(f"  {d:<22s} {by_dim_count[d]:4d}", file=sys.stderr)
    if missing_rubric:
        print(f"[stage1] dims missing rubric: {sorted(missing_rubric)}", file=sys.stderr)
    if skipped_excluded:
        print(f"[stage1] dims excluded by --exclude_dims: "
              f"{dict(skipped_excluded)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair_glob",
                   default="data/v_arr_mvp/*_mvp/v2_prompt_pair.json",
                   help="glob (relative to repo root) of per-dim pair files")
    p.add_argument("--rubric_dir",
                   default="rubric_pipeline/rubrics",
                   help="dir containing v_arr_mvp_<dim>_v2.txt scaffolds")
    p.add_argument("--output",
                   default="data/rm_train/gold_rubrics_18d.jsonl")
    p.add_argument("--split_outdir",
                   default="data/rm_train/splits",
                   help="dir for the 90/10 train.jsonl / val.jsonl split")
    p.add_argument("--val_frac",     type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=0)
    # Cache convention identical to stage0/build_pair_data.py.
    p.add_argument("--baseline_cache_dir",
                   default="runs/baselines/_video_cache")
    p.add_argument("--prefetch_cache_dir",
                   default="data/v_arr_mvp/cache_videos")
    p.add_argument("--require_local", action="store_true",
                   help="drop pairs whose videos can't be resolved locally")
    p.add_argument("--exclude_dims", nargs="+", default=["soundscape"],
                   help="dim slugs to skip entirely. Default ['soundscape']: "
                        "Qwen2.5-VL is vision-only and can't hear audio, so "
                        "the soundscape dim is structurally untrainable on "
                        "this backbone (stage0 D6 confirmed PLCC ≈ 0). "
                        "Pass --exclude_dims with no args (or '') to keep "
                        "all 18 dims for ablation.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    build(args)


if __name__ == "__main__":
    main()
