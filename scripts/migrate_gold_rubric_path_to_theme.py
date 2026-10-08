"""Rename ``"path"`` → ``"theme"`` in every rubric of ``gold_rubrics_18d.clean.jsonl``.

Backward-compat parser (``rm_training.stage2_grpo.rubric_parser`` and
``rm_training.stage2_grpo.reward_fn._r_align``) still accepts the legacy
``"path"`` key, so this migration is a cosmetic alignment — runtime behaviour
is unchanged whether you run it or not. We keep it under VCS so production
data lands on the canonical schema and matches what the model is now taught
to emit (``RUBRIC_PROMPT_TMPL`` uses ``"theme"``).

Usage::

    cd /workspace/user_code/40063/jacuzzitang/projects/RewardVerse
    python scripts/migrate_gold_rubric_path_to_theme.py \\
        --input  data/rm_train/gold_rubrics_18d.clean.jsonl \\
        --output data/rm_train/gold_rubrics_18d.clean.jsonl

Input and output may be the same file. A timestamped backup is written next
to the input before any in-place rewrite.

The transformation:
  - For each line (JSON object), look for ``gold_rubric: list[dict]``.
  - For each entry in that list, if ``"path"`` is present and ``"theme"``
    is NOT, rename ``"path"`` to ``"theme"`` (preserving the value verbatim,
    so the ``"Theme: ..."`` prefix gold uses survives).
  - Lines without a ``gold_rubric`` field are passed through unchanged.

The script is idempotent: running it twice does nothing on the second pass.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def migrate_record(rec: dict) -> tuple[dict, int]:
    """Return a new record with rubric ``path`` keys renamed to ``theme``.

    Returns ``(rec_out, n_renamed)``. n_renamed counts how many entries
    inside the record's ``gold_rubric`` list got their key renamed.
    """
    gold = rec.get("gold_rubric")
    if not isinstance(gold, list):
        return rec, 0
    n_renamed = 0
    new_gold = []
    for entry in gold:
        if not isinstance(entry, dict):
            new_gold.append(entry)
            continue
        if "path" in entry and "theme" not in entry:
            new_entry = {}
            for k, v in entry.items():
                new_entry["theme" if k == "path" else k] = v
            new_gold.append(new_entry)
            n_renamed += 1
        else:
            new_gold.append(entry)
    rec_out = dict(rec)
    rec_out["gold_rubric"] = new_gold
    return rec_out, n_renamed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  required=True, help="path to gold_rubrics_*.jsonl")
    p.add_argument("--output", required=True, help="output jsonl path (may equal --input)")
    p.add_argument("--no-backup", action="store_true",
                   help="skip the timestamped backup of the input file (use only "
                        "if you have your own VCS / backup)")
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.is_file():
        print(f"[migrate] input not found: {src}", file=sys.stderr)
        return 2

    if dst.resolve() == src.resolve() and not args.no_backup:
        ts = time.strftime("%Y%m%dT%H%M%S")
        backup = src.with_suffix(src.suffix + f".bak.{ts}")
        shutil.copy2(src, backup)
        print(f"[migrate] backup: {backup}", file=sys.stderr)

    n_lines = 0
    n_records_changed = 0
    n_entries_renamed = 0
    out_lines: list[str] = []
    with src.open() as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                out_lines.append("")
                continue
            n_lines += 1
            rec = json.loads(ln)
            rec_out, n = migrate_record(rec)
            if n > 0:
                n_records_changed += 1
                n_entries_renamed += n
            out_lines.append(json.dumps(rec_out, ensure_ascii=False))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(
        f"[migrate] processed {n_lines} records, "
        f"changed {n_records_changed}, renamed {n_entries_renamed} rubric entries "
        f"(path → theme)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
