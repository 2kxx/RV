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
2. ``GoldRubricDataset``                     — torch-style ``__getitem__``
3. ``collate_pair_for_grpo(rows) -> dict``   — minimal field repack

Intentionally minimal so train_grpo.py can plug new fields (caption,
domain id, etc.) without touching this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "load_jsonl",
    "GoldRubricDataset",
    "collate_pair_for_grpo",
]


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
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
    def from_jsonl(cls, path: str | Path) -> "GoldRubricDataset":
        return cls(rows=load_jsonl(path))

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
            "gold_rubric", "gt_label"]
    out: dict[str, list] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            out[k].append(r.get(k, "" if k != "gold_rubric" else []))
    return out
