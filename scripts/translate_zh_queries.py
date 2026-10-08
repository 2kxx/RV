"""Translate Chinese ``query`` fields to English using gemini-3-flash via
the Venus llmproxy.

Supports two input layouts:

* **JSONL** (one dict per line) — original use case, e.g.
  ``data/rm_train/gold_rubrics_18d.clean.jsonl``.
* **JSON list** (a top-level list of dicts in one file) — e.g. the
  per-dim ``data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json``
  evaluation sets. Accepts ``--inputs`` to translate many such files
  in one run, sharing the cache and worker pool across them.

Why this script exists
----------------------
The 18-dim gold rubric set was built from a mix of Chinese and English
v_arr_mvp pair files. After ``rebuild_gold_rubrics_query.py`` ran, ~149
rows still carry Chinese (or Chinese-mixed) captions in ``query``. That
language imbalance is one of the suspected contributors to the train_joint
collapse at step 30 (the model fell into a Chinese/English mixed
auto-regressive loop in its rubric output by step 90, see
``logs/train_joint.log``). Translating them aligns the whole corpus to
English so the student doesn't get pushed off-distribution by a minority
of zh rows.

The same language mismatch was also the cause of the post-2026-06-21
PLCC regression: training data is all English, but the test_pointwise
files reverted to Chinese ``query`` strings, so the score head ran on
zh prompts at eval time. Translating the test set back to English with
this script restores parity. The original zh string is left intact in
``metadata.prompt_zh`` (already present in those files) for traceability.

Behaviour
---------
* Reads each row; rows tagged ``no_query: true`` are skipped (their
  caption is already an English template).
* Rows with no Chinese characters are skipped.
* Rows with Chinese (zh-only or zh+en mix) are translated; the system
  prompt instructs gemini to:
    - preserve role tokens like ``Person_01``, ``Soldier_02`` verbatim,
    - preserve shot markers like ``[shot_0_prompt]``,
    - return ONLY the translated text (no preamble / no quotes),
    - keep the same paragraph / line structure.
* Translations are cached in a sidecar JSON keyed by (row-id, sha1) so
  reruns are free and a query edit invalidates its stale entry. Pass
  ``--force`` to ignore the cache. Row-id falls back to ``id`` then
  ``pair_id`` then ``"<file>:<idx>"``.
* Parallel API calls via thread-pool (default 8 workers). The proxy
  endpoint is sync HTTP so threads are sufficient. The cache is flushed
  to disk every ``--cache-flush-every`` completions so a crash mid-batch
  only loses in-flight calls; rerunning resumes from cache.
* On failure (3 retries with exponential backoff), the original Chinese
  is kept and the row id is logged; the script continues so one bad
  row doesn't kill the whole batch.

Usage
-----
JSONL (single file, in-place):

::

    python scripts/translate_zh_queries.py \\
        --input  data/rm_train/gold_rubrics_18d.clean.jsonl \\
        --output data/rm_train/gold_rubrics_18d.clean.jsonl \\
        --cache  data/rm_train/.translate_cache.json \\
        --workers 8

JSON list (multi-file, in-place across every test_pointwise dim):

::

    python scripts/translate_zh_queries.py \\
        --format json-list \\
        --inputs data/v_arr_mvp/*_mvp/test_pointwise_*.json \\
        --cache  data/v_arr_mvp/.translate_cache.json \\
        --workers 8

Pass ``--dry-run`` to see what would be translated without calling the
API or writing the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types

API_KEY = "d0fdEk0UAehNrBhflbV2etjU@3797"
BASE_URL = "http://v2.open.venus.oa.com/llmproxy/"
MODEL = "gemini-3-flash"

CHN = re.compile(r"[一-鿿]")

SYSTEM_PROMPT = (
    "You are a professional zh→en translator for video-generation "
    "captions. Translate the user's Chinese (or Chinese/English mixed) "
    "text into fluent, natural English, keeping the cinematic register. "
    "STRICT rules:\n"
    " 1. Output ONLY the English translation. No preamble, no quotes, "
    "    no notes, no markdown.\n"
    " 2. Preserve role tokens like 'Person_01', 'Soldier_02', "
    "    'Other_Soldiers' EXACTLY as written. Do not translate them.\n"
    " 3. Preserve shot markers like '[shot_0_prompt]' EXACTLY.\n"
    " 4. Preserve paragraph and line breaks (\\n) from the input.\n"
    " 5. If the input is already English, return it unchanged.\n"
    " 6. Never refuse, never add commentary."
)


def make_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        http_options=types.HttpOptionsDict(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            base_url_resource_scope=types.ResourceScope.COLLECTION,
        ),
    )


def translate_one(client: genai.Client, text: str, *, max_retries: int = 3) -> str:
    """Single API call with exponential-backoff retry. Raises on final failure."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            out = (resp.text or "").strip()
            if not out:
                raise RuntimeError("empty response")
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"translate failed after {max_retries} retries: {last_err}")


def needs_translation(query: str) -> bool:
    return bool(CHN.search(query))


def cache_key(row_id: str, src: str) -> str:
    h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]
    return f"{row_id}::{h}"


def _row_id(row: dict, fallback: str) -> str:
    """Row identifier for cache + logging. Prefer ``id``, then ``pair_id``,
    finally a positional fallback like ``"<file>:<idx>"``.
    """
    for k in ("id", "pair_id"):
        v = row.get(k)
        if v:
            return str(v)
    return fallback


# ---------------------------------------------------------------------------
# I/O — two layouts: JSONL and JSON list. Both yield (rows, write_back).
# ---------------------------------------------------------------------------
def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _load_json_list(p: Path) -> list[dict]:
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"{p}: expected a top-level JSON list, got {type(data).__name__}"
        )
    return data


def _write_json_list(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    tmp.replace(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--format",
        choices=("jsonl", "json-list"),
        default="jsonl",
        help="Input layout: 'jsonl' (one dict per line, e.g. gold_rubrics)"
             " or 'json-list' (top-level list of dicts, e.g. test_pointwise).",
    )
    # JSONL layout uses --input/--output (single file). JSON-list layout
    # uses --inputs (multiple files, each written back in place); the two
    # are mutually exclusive in practice but argparse-wise both are
    # accepted and validated below.
    ap.add_argument("--input", type=Path, default=None,
                    help="JSONL input file (used when --format jsonl).")
    ap.add_argument("--output", type=Path, default=None,
                    help="JSONL output file (used when --format jsonl). "
                         "Pass the same path as --input for in-place update.")
    ap.add_argument("--inputs", nargs="+", type=Path, default=None,
                    help="JSON-list input files (used when --format "
                         "json-list). Each file is rewritten in place; "
                         "the cache is shared across all of them.")
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("data/rm_train/.translate_cache.json"),
        help="JSON sidecar mapping (row_id, sha1) -> translated text",
    )
    ap.add_argument(
        "--cache-flush-every",
        type=int,
        default=10,
        help="flush cache to disk every N successful translations",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of concurrent translation threads",
    )
    ap.add_argument("--force", action="store_true",
                    help="ignore cache, re-translate everything")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ---- Resolve inputs into a list of (file_path, rows) buckets so the
    # same loop handles 1×JSONL or N×JSON-list uniformly.
    buckets: list[tuple[Path, list[dict], str]] = []  # (path, rows, layout)
    if args.format == "jsonl":
        if args.input is None or args.output is None:
            ap.error("--format jsonl requires --input and --output")
        if args.inputs:
            ap.error("--inputs is only valid with --format json-list")
        rows = _load_jsonl(args.input)
        print(f"[translate] loaded {len(rows)} rows from {args.input}")
        buckets.append((args.output, rows, "jsonl"))
        # Note: we keep the read-from/write-to split for JSONL (legacy
        # behaviour), so the file referenced in 'buckets' is the OUTPUT.
        # For JSON-list we always rewrite in place.
    else:
        if not args.inputs:
            ap.error("--format json-list requires --inputs FILE [FILE ...]")
        if args.input or args.output:
            ap.error("--input/--output are only valid with --format jsonl")
        for p in args.inputs:
            rows = _load_json_list(p)
            print(f"[translate] loaded {len(rows)} rows from {p}")
            buckets.append((p, rows, "json-list"))

    # ---- Walk every bucket, collect rows that need translation.
    # todo entries: (bucket_idx, row_idx, row_id, query)
    todo: list[tuple[int, int, str, str]] = []
    skipped_no_query = 0
    skipped_english = 0
    for b_idx, (p, rows, _layout) in enumerate(buckets):
        for i, r in enumerate(rows):
            if r.get("no_query"):
                skipped_no_query += 1
                continue
            q = r.get("query", "")
            if not needs_translation(q):
                skipped_english += 1
                continue
            rid = _row_id(r, fallback=f"{p.name}:{i}")
            todo.append((b_idx, i, rid, q))

    print(
        f"[translate] to_translate={len(todo)}, "
        f"skipped_no_query={skipped_no_query}, "
        f"skipped_english={skipped_english}"
    )

    # ---- Cache load
    cache: dict[str, str] = {}
    if args.cache.exists() and not args.force:
        try:
            cache = json.loads(args.cache.read_text())
            print(f"[translate] loaded cache with {len(cache)} entries")
        except Exception as e:  # noqa: BLE001
            print(f"[translate] cache unreadable ({e}); starting fresh")

    # ---- Apply cache hits up front, collect misses to call API for.
    hits = 0
    # pending entries: (bucket_idx, row_idx, row_id, src, cache_key)
    pending: list[tuple[int, int, str, str, str]] = []
    for b_idx, r_idx, rid, src in todo:
        key = cache_key(rid, src)
        if key in cache:
            buckets[b_idx][1][r_idx]["query"] = cache[key]
            hits += 1
        else:
            pending.append((b_idx, r_idx, rid, src, key))
    print(
        f"[translate] cache hits: {hits}/{len(todo)}, "
        f"need API calls: {len(pending)}"
    )

    if args.dry_run:
        print("[translate] --dry-run: not calling API, not writing output.")
        return

    # ---- Concurrent translation
    failures: list[tuple[str, str]] = []  # (row_id, error)
    if pending:
        client = make_client()
        print(
            f"[translate] starting {args.workers} workers for "
            f"{len(pending)} translations..."
        )
        cache_lock = threading.Lock()
        progress_lock = threading.Lock()
        done_count = [0]
        last_flush = [time.time()]

        def work(item):
            b_idx, r_idx, rid, src, key = item
            try:
                out = translate_one(client, src)
            except Exception as e:  # noqa: BLE001
                with progress_lock:
                    done_count[0] += 1
                    n = done_count[0]
                print(f"  [{n}/{len(pending)}] {rid} FAILED: {e}", flush=True)
                return (rid, str(e))

            with cache_lock:
                cache[key] = out
            buckets[b_idx][1][r_idx]["query"] = out

            with progress_lock:
                done_count[0] += 1
                n = done_count[0]
                should_flush = (
                    n % args.cache_flush_every == 0
                    or time.time() - last_flush[0] > 30
                )
            print(
                f"  [{n}/{len(pending)}] {rid} ok "
                f"({len(src)}→{len(out)} chars)",
                flush=True,
            )
            if should_flush:
                with cache_lock:
                    args.cache.parent.mkdir(parents=True, exist_ok=True)
                    args.cache.write_text(
                        json.dumps(cache, ensure_ascii=False, indent=2)
                    )
                last_flush[0] = time.time()
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, item) for item in pending]
            for fut in as_completed(futures):
                err = fut.result()
                if err:
                    failures.append(err)

        # Final cache flush
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        print(f"[translate] cache saved -> {args.cache}")

    if failures:
        print(f"[translate] FAILURES: {len(failures)}", file=sys.stderr)
        for rid, err in failures[:10]:
            print(f"   {rid}: {err}", file=sys.stderr)
        if len(failures) > 10:
            print(f"   ...and {len(failures) - 10} more", file=sys.stderr)

    # ---- Write outputs
    total_written = 0
    for (p, rows, layout) in buckets:
        if layout == "jsonl":
            _write_jsonl(p, rows)
        else:
            _write_json_list(p, rows)
        total_written += len(rows)
        print(f"[translate] wrote {len(rows)} rows -> {p}")

    # Final audit
    remaining = 0
    for (_p, rows, _layout) in buckets:
        for r in rows:
            if not r.get("no_query") and needs_translation(r.get("query", "")):
                remaining += 1
    print(f"[translate] rows still containing Chinese: {remaining}")


if __name__ == "__main__":
    main()
