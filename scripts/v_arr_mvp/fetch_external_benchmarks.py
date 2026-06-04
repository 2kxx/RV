"""RewardVerse — external benchmark fetcher (VideoGen-RewardBench).

Builds a pairwise JSON subset (or the full eval split) of
KlingAIResearch/VideoGen-RewardBench that downstream eval scripts
(rubric / VideoScore / VideoScore2) can ingest with **no schema bridge**.

Single subcommand, kept as a subparser for backward CLI compatibility:

  videogen_rewardbench
      KlingAIResearch/VideoGen-RewardBench (HF, split='eval').
      Emits a *pairwise* JSON whose every record carries A/B paths
      and the four VGRB pairwise labels (VQ / MQ / TA / Overall in
      {'A', 'B', 'same'}).
      Default: emit the FULL eval split (~25.2k pairs). Pass
      ``--num_pairs N`` to keep stratified sampling for a smoke run.

Behaviour:
  * Re-uses the local snapshot under
    ``/group/40063/jacuzzitang/datasets/<short_name>/`` populated by
    ``local_assets.ensure_dataset_local``.
  * Selectively unzips ONLY the videos referenced by the chosen subset
    (when sampling). When fetching the full split, unzips all referenced
    paths (still selective vs. the giant videos.zip).
  * Writes all artefacts under the user-chosen ``--out_dir`` (default
    ``/group/40063/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench/``).

Note:
  The previous ``videofeedback`` subcommand has been **removed** —
  RewardVerse no longer evaluates on the VideoFeedback-200 subset.
  Use this script only for VGRB.

Usage::

    # Full VGRB eval split (default)
    python scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench

    # Smoke / dev sample (stratified 200 pairs)
    python scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench \\
        --num_pairs 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_assets import (  # type: ignore  # noqa: E402
    ensure_dataset_local,
)


# ===========================================================================
# Shared helpers.
# ===========================================================================
def _load_parquet_rows(parquet_path: Path) -> list[dict]:
    """Read a parquet shard into a list[dict] using pyarrow (no streaming)."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required: pip install pyarrow"
        ) from exc
    table = pq.read_table(str(parquet_path))
    return table.to_pylist()


def _stratified_round_robin(
    items: list[Any],
    bucket_fn,
    target: int,
    seed: int,
) -> list[Any]:
    """Round-robin sample ``target`` items across the buckets returned
    by ``bucket_fn(item)``. Deterministic given ``seed``."""
    rng = random.Random(seed)
    buckets: dict[Any, list[Any]] = defaultdict(list)
    for it in items:
        buckets[bucket_fn(it)].append(it)
    keys = sorted(buckets.keys(), key=lambda k: (str(k),))
    for k in keys:
        rng.shuffle(buckets[k])
    chosen: list[Any] = []
    cursors = {k: 0 for k in keys}
    while len(chosen) < target:
        progressed = False
        for k in keys:
            if len(chosen) >= target:
                break
            if cursors[k] < len(buckets[k]):
                chosen.append(buckets[k][cursors[k]])
                cursors[k] += 1
                progressed = True
        if not progressed:
            break
    return chosen


def _selective_unzip(
    zip_path: Path,
    members: Iterable[str],
    out_dir: Path,
    *,
    flatten: bool = True,
) -> dict[str, Path]:
    """Extract ONLY the named members from ``zip_path`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    members_set = set(members)
    if not members_set:
        return extracted

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        resolved: dict[str, str] = {}
        for m in members_set:
            if m in names:
                resolved[m] = m
                continue
            for cand in (
                f"videos_annotated_train/{m}",
                f"videos_annotated_test/{m}",
                f"videos/{m}",
            ):
                if cand in names:
                    resolved[m] = cand
                    break
        missing = members_set - set(resolved.keys())
        if missing:
            print(f"[warn] {len(missing)} members not found in {zip_path.name}; "
                  f"first few: {sorted(missing)[:3]}", file=sys.stderr)

        for i, (logical, actual) in enumerate(resolved.items()):
            target = out_dir / (Path(logical).name if flatten else logical)
            if target.exists() and target.stat().st_size > 0:
                extracted[logical] = target
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(actual) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted[logical] = target
            if (i + 1) % 100 == 0 or (i + 1) == len(resolved):
                print(f"[unzip] {i + 1}/{len(resolved)} extracted "
                      f"({zip_path.name})", flush=True)
    return extracted


# ===========================================================================
# VideoGen-RewardBench pairwise.
# ===========================================================================
# Confirmed by inspect_external_schema.py on
# KlingAIResearch/VideoGen-RewardBench (split='eval').
_VGRB_PROMPT_KEY = "prompt"
_VGRB_PATH_A_KEY = "path_A"
_VGRB_PATH_B_KEY = "path_B"
_VGRB_LABEL_KEYS = {"VQ", "MQ", "TA", "Overall"}  # values in {'A','B','same'}

# VGRB pairwise axis -> RewardVerse 18-dim axis mapping.
_VGRB_TO_REWARDVERSE: dict[str, list[str]] = {
    "VQ":      ["visual_quality"],
    "MQ":      ["action"],
    "TA":      ["character", "scene"],
    "Overall": [],   # special: aggregate of all rubric axes downstream
}


def _vgrb_load_rows(snap_dir: Path) -> list[dict]:
    """Load VGRB rows. Strategy: prefer the parquet under the HF
    snapshot if present, else fall back to the CSV that VGRB ships."""
    for cand in snap_dir.glob("*.parquet"):
        try:
            return _load_parquet_rows(cand)
        except Exception as exc:
            print(f"[warn] failed to load {cand}: {exc}", file=sys.stderr)

    csv_path = snap_dir / "videogen-rewardbench.csv"
    if csv_path.exists():
        import csv as _csv
        with csv_path.open("r", encoding="utf-8") as f:
            return [dict(r) for r in _csv.DictReader(f)]

    raise FileNotFoundError(
        f"no parquet or videogen-rewardbench.csv found under {snap_dir}"
    )


def cmd_videogen_rewardbench(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"

    snap_dir = ensure_dataset_local(
        "KlingAIResearch/VideoGen-RewardBench",
        allow_download=not args.no_download,
    )
    print(f"[VGRB] snapshot dir: {snap_dir}", flush=True)
    rows = _vgrb_load_rows(snap_dir)
    print(f"[VGRB] loaded {len(rows)} rows", flush=True)

    # 1) Validate. Keep only rows with both paths + non-trivial prompt + the
    #    4 pairwise labels.
    def _norm_label(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"a"}:    return "A"
        if s in {"b"}:    return "B"
        if s in {"same", "tie", "equal"}: return "same"
        return None

    pool: list[dict] = []
    for r in rows:
        prompt = r.get(_VGRB_PROMPT_KEY)
        pa     = r.get(_VGRB_PATH_A_KEY)
        pb     = r.get(_VGRB_PATH_B_KEY)
        if not (prompt and pa and pb):
            continue
        labels: dict[str, str] = {}
        ok = True
        for k in _VGRB_LABEL_KEYS:
            v = _norm_label(r.get(k))
            if v is None:
                ok = False
                break
            labels[k] = v
        if not ok:
            continue
        pool.append({
            "prompt":  str(prompt),
            "path_A":  str(pa),
            "path_B":  str(pb),
            "labels":  labels,
            "A_model": str(r.get("A_model", "?")),
            "B_model": str(r.get("B_model", "?")),
        })
    print(f"[VGRB] {len(pool)} pairs pass schema validation", flush=True)

    # 2) Decide sampling: if --num_pairs is None or <=0, use the full pool.
    use_full = (args.num_pairs is None) or (args.num_pairs <= 0) \
               or (args.num_pairs >= len(pool))
    if use_full:
        chosen = list(pool)
        print(f"[VGRB] FULL eval split: {len(chosen)} pairs", flush=True)
    else:
        def _bucket(it: dict) -> tuple[str, frozenset[str]]:
            return (
                it["labels"]["Overall"],
                frozenset({it["A_model"], it["B_model"]}),
            )
        chosen = _stratified_round_robin(
            pool, bucket_fn=_bucket, target=args.num_pairs, seed=args.seed,
        )
        print(f"[VGRB] sampled {len(chosen)} pairs (target={args.num_pairs})",
              flush=True)

    # 3) Selectively unzip the 2*N videos referenced.
    zip_path = snap_dir / "videos.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"VGRB videos.zip not found at {zip_path}"
        )
    members: list[str] = []
    seen: set[str] = set()
    for c in chosen:
        for p in (c["path_A"], c["path_B"]):
            if p not in seen:
                members.append(p)
                seen.add(p)
    print(f"[VGRB] selectively unzipping {len(members)} unique videos "
          f"-> {videos_dir}", flush=True)
    extracted = _selective_unzip(
        zip_path, members, videos_dir, flatten=False,
    )

    # 4) Emit pairwise JSON.
    payload: list[dict] = []
    overall_count: Counter = Counter()
    for i, c in enumerate(chosen):
        overall_count[c["labels"]["Overall"]] += 1
        local_a = extracted.get(c["path_A"])
        local_b = extracted.get(c["path_B"])
        if local_a is None or local_b is None:
            print(f"[warn] pair {i} missing local mp4(s); skipping",
                  file=sys.stderr)
            continue
        rec: dict[str, Any] = {
            "id":          f"vgrb_{i:05d}",
            "query":       c["prompt"],
            "video_a":     str(local_a),
            "video_b":     str(local_b),
            "A_model":     c["A_model"],
            "B_model":     c["B_model"],
            "labels":      c["labels"],     # {VQ, MQ, TA, Overall} -> {A,B,same}
            "rewardverse_axis_mapping": _VGRB_TO_REWARDVERSE,
            "task_type":   "t2v_pairwise",
            "metadata": {
                "source": "KlingAIResearch/VideoGen-RewardBench",
                "split":  "eval",
            },
        }
        payload.append(rec)

    out_json = out_dir / f"vgrb_pairwise_{len(payload)}.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[VGRB] wrote {len(payload)} pairs -> {out_json}", flush=True)
    print(f"[VGRB] Overall label distribution: {dict(overall_count)}",
          flush=True)


# ===========================================================================
# CLI.
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    DEFAULT_OUT_ROOT = Path(
        "/group/40063/jacuzzitang/datasets/_v_arr_external_subsets"
    )

    p_vg = sub.add_parser(
        "videogen_rewardbench",
        help="Build a VideoGen-RewardBench pairwise JSON. "
             "Default = full eval split (~25.2k pairs). "
             "Use --num_pairs N for a stratified subsample.",
    )
    # num_pairs default is None => emit the FULL eval split.
    p_vg.add_argument("--num_pairs", type=int, default=None,
                      help="If unset or <=0, emit the FULL VGRB eval split.")
    p_vg.add_argument("--seed", type=int, default=0,
                      help="Seed for stratified subsampling (ignored on full split).")
    p_vg.add_argument("--out_dir", type=str,
                      default=str(DEFAULT_OUT_ROOT / "videogen_rewardbench"))
    p_vg.add_argument("--no_download", action="store_true")
    p_vg.set_defaults(func=cmd_videogen_rewardbench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()