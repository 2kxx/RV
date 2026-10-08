"""Rebuild the ``query`` field of ``gold_rubrics_18d.clean.jsonl``.

Two transforms are applied in order, both guarded by ``pair_id`` lookups
(no positional / index-based reasoning, so re-running on a partially
transformed file is a no-op):

1. **Restore placeholder queries.**
   Rows in dimensions {consistency, logic, progression} were built with a
   constant placeholder string::

       "(no prompt supplied; rank the two videos purely by intrinsic
        visual quality)"

   The real Chinese prompt for each pair lives in
   ``data/v_arr_mvp/{dim}_mvp/v2_prompt_pair.json`` under
   ``metadata.prompt_zh`` (verified 30/30 for each of the 3 dims). We
   pull the prompt back in via ``pair_id`` match.

2. **Sample N no-query rows uniformly across all dimensions.**
   "No-query" here means "no video-generation prompt was provided", NOT
   "empty string". The student still needs to know which dimension to
   produce rubrics for, so we write a short instructional caption into
   ``query`` that names the dimension. Default template::

       Please evaluate the {dimension} dimension based on the videos'
       intrinsic visual content.

   Override with ``--no-query-template``. The sample is stratified
   per-dimension (90 / 16 dims ≈ 5–6 per dim, with the remainder spread
   across the first few dims by sorted name) so no single dim is wiped.

   We tag the sampled rows with ``no_query: true`` so downstream code or
   audits can identify them. A fixed seed makes the choice reproducible.

Usage
-----
::

    python scripts/rebuild_gold_rubrics_query.py \
        --input  data/rm_train/gold_rubrics_18d.clean.jsonl \
        --output data/rm_train/gold_rubrics_18d.clean.jsonl \
        --no-query-count 90 \
        --seed 42

Pass ``--dry-run`` to print the audit without writing.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PLACEHOLDER = (
    "(no prompt supplied; rank the two videos purely by intrinsic "
    "visual quality)"
)
PLACEHOLDER_DIMS = ("consistency", "logic", "progression")

# Used when a row is marked no-query: the student still needs to know
# which dimension to score. ``{dimension}`` is filled per-row.
NO_QUERY_TEMPLATE = (
    "Please evaluate the {dimension} dimension based on the videos' "
    "intrinsic visual content."
)


def load_prompt_zh_map(repo_root: Path) -> dict[str, str]:
    """{pair_id: prompt_zh} for the 3 source files that carry placeholders."""
    out: dict[str, str] = {}
    for dim in PLACEHOLDER_DIMS:
        src = repo_root / f"data/v_arr_mvp/{dim}_mvp/v2_prompt_pair.json"
        rows = json.loads(src.read_text())
        for r in rows:
            m = r["metadata"]
            if isinstance(m, str):
                m = ast.literal_eval(m)
            pz = (m.get("prompt_zh") or "").strip()
            if pz:
                out[r["id"]] = pz
    return out


def stratified_sample_indices(
    rows: list[dict],
    n: int,
    rng: random.Random,
) -> set[int]:
    """Pick ``n`` row indices stratified across ``dimension``.

    Each dim contributes ``floor(n / D)`` rows; the remainder is sprinkled
    across the dims that come first by sorted name (deterministic given seed
    only via ``rng.shuffle`` of within-dim picks — the dim order itself is
    sorted-stable).
    """
    by_dim: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_dim[r["dimension"]].append(i)

    dims = sorted(by_dim)
    D = len(dims)
    per_dim_base, leftover = divmod(n, D)

    chosen: set[int] = set()
    for k, dim in enumerate(dims):
        take = per_dim_base + (1 if k < leftover else 0)
        pool = list(by_dim[dim])
        rng.shuffle(pool)
        if take > len(pool):
            print(
                f"[rebuild] WARNING: dim={dim} has only {len(pool)} rows "
                f"but stratified sample wants {take}; taking all of them.",
                file=sys.stderr,
            )
            take = len(pool)
        chosen.update(pool[:take])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root (for resolving data/v_arr_mvp/...)",
    )
    ap.add_argument(
        "--no-query-count",
        type=int,
        default=90,
        help="number of rows to mark as no-query (query becomes the "
             "dimension-only template) after restoration",
    )
    ap.add_argument(
        "--no-query-template",
        type=str,
        default=NO_QUERY_TEMPLATE,
        help="format string for the no-query caption; must contain "
             "{dimension}",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows: list[dict] = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[rebuild] loaded {len(rows)} rows from {args.input}")

    # ---- Step 1: restore placeholder queries.
    prompt_zh = load_prompt_zh_map(args.repo_root)
    restored = 0
    missing: list[str] = []
    for r in rows:
        if r["dimension"] not in PLACEHOLDER_DIMS:
            continue
        if r.get("query", "").strip() != PLACEHOLDER:
            continue
        pz = prompt_zh.get(r["pair_id"], "")
        if pz:
            r["query"] = pz
            restored += 1
        else:
            missing.append(r["pair_id"])
    print(f"[rebuild] restored {restored} placeholder queries from prompt_zh")
    if missing:
        print(
            f"[rebuild] WARNING: {len(missing)} placeholder rows had no "
            f"matching prompt_zh; they keep the placeholder. Examples: "
            f"{missing[:5]}",
            file=sys.stderr,
        )

    # ---- Step 2: stratified no-query sample.
    # Idempotency: if rows are ALREADY tagged ``no_query: true`` we honour
    # those tags rather than re-sampling, so re-running with a different
    # template just rewrites their captions in-place (no churn in which
    # rows are no-query).
    if "{dimension}" not in args.no_query_template:
        sys.exit("--no-query-template must contain '{dimension}'")

    existing_no_query = {i for i, r in enumerate(rows) if r.get("no_query") is True}
    if existing_no_query:
        no_query_idx = existing_no_query
        print(
            f"[rebuild] honouring existing no_query=True tags on "
            f"{len(no_query_idx)} rows (idempotent re-run)."
        )
    else:
        rng = random.Random(args.seed)
        no_query_idx = stratified_sample_indices(rows, args.no_query_count, rng)
        print(
            f"[rebuild] sampled {len(no_query_idx)} no-query rows "
            f"(seed={args.seed}, target={args.no_query_count})"
        )

    nq_dim_counter: Counter[str] = Counter()
    for i in no_query_idx:
        rows[i]["query"] = args.no_query_template.format(
            dimension=rows[i]["dimension"]
        )
        rows[i]["no_query"] = True
        nq_dim_counter[rows[i]["dimension"]] += 1
    for i, r in enumerate(rows):
        if i not in no_query_idx:
            r["no_query"] = False

    print("[rebuild] no-query distribution by dimension:")
    for dim in sorted(nq_dim_counter):
        print(f"           {dim:<18} {nq_dim_counter[dim]}")

    # ---- Audit summary
    final_no_query = sum(1 for r in rows if r.get("no_query") is True)
    final_placeholder = sum(
        1 for r in rows if r.get("query", "").strip() == PLACEHOLDER
    )
    print(
        f"[rebuild] FINAL: no_query=True: {final_no_query}, "
        f"old-placeholder remaining: {final_placeholder}"
    )

    if args.dry_run:
        print("[rebuild] --dry-run: not writing output.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(f"[rebuild] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
