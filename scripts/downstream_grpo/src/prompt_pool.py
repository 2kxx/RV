"""Prompt-pool loader for downstream GRPO.

Reads the existing English-language EvalVerse training corpus
`data/rm_train/gold_rubrics_18d.jsonl` and produces three stratified
splits (train / eval / fixed32) that share NO prompts.

Design notes
------------
* We consume ONLY `query` and `dimension` from each entry. The video paths
  in `y_pos`/`y_neg` were for RM training and are irrelevant here.
* Split is deterministic given `seed` for cross-arm reproducibility.
* Every downstream arm sees the same train/eval/fixed32 partition.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Prompt:
    prompt_id: str          # dim + running index, e.g. "action-004"
    query: str              # English caption
    dimension: str          # EvalVerse dim name


@dataclass
class PromptSplits:
    train: list[Prompt]
    eval: list[Prompt]
    fixed32: list[Prompt]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt_pool(
    path: str | Path,
    *,
    dims_whitelist: Sequence[str],
    train_size: int,
    eval_size: int,
    fixed_size: int,
    seed: int = 42,
) -> PromptSplits:
    """Load and split the EvalVerse prompt pool.

    Records are deduplicated by (dimension, query). If the whitelist covers
    16 dims and the corpus has ~40 pairs/dim, we typically get 640 unique
    prompts, more than enough for the requested splits.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    records = _load_jsonl(path)

    whitelist = set(dims_whitelist)
    by_dim: dict[str, list[Prompt]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for rec in records:
        dim = rec.get("dimension")
        query = rec.get("query")
        if not dim or not query:
            continue
        if dim not in whitelist:
            continue
        key = (dim, query.strip())
        if key in seen:
            continue
        seen.add(key)
        idx = len(by_dim[dim])
        by_dim[dim].append(Prompt(f"{dim}-{idx:04d}", query.strip(), dim))

    rng = random.Random(seed)
    for dim in by_dim:
        rng.shuffle(by_dim[dim])

    # Priority: fixed32 first, then eval, then train — so the eval/fixed
    # sets always exist even when the corpus is small (the EvalVerse
    # training corpus has ~14-26 unique prompts per dim, ~371 total).
    n_dims = len(whitelist)
    per_dim_fixed = max(1, fixed_size // n_dims)
    per_dim_eval = max(1, eval_size // n_dims)
    # train uses whatever is left per dim (no per-dim cap).

    train: list[Prompt] = []
    eval_: list[Prompt] = []
    fixed: list[Prompt] = []
    for dim in sorted(whitelist):
        pool = by_dim.get(dim, [])
        if not pool:
            print(f"[prompt_pool] WARN: dim {dim} has 0 prompts in corpus")
            continue
        cursor = 0

        take = min(per_dim_fixed, len(pool) - cursor)
        fixed.extend(pool[cursor:cursor + take])
        cursor += take

        take = min(per_dim_eval, len(pool) - cursor)
        eval_.extend(pool[cursor:cursor + take])
        cursor += take

        # everything else goes to train
        train.extend(pool[cursor:])

    # Trim to requested sizes; already stratified.
    rng.shuffle(train)
    rng.shuffle(eval_)
    rng.shuffle(fixed)
    return PromptSplits(
        train=train[:train_size],
        eval=eval_[:eval_size],
        fixed32=fixed[:fixed_size],
    )


def write_split(prompts: Sequence[Prompt], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({
                "prompt_id": p.prompt_id,
                "query": p.query,
                "dimension": p.dimension,
            }, ensure_ascii=False) + "\n")


def read_split(path: str | Path) -> list[Prompt]:
    return [
        Prompt(rec["prompt_id"], rec["query"], rec["dimension"])
        for rec in _load_jsonl(Path(path))
    ]


class PromptIterator:
    """Infinite shuffled iterator over a prompt list (for the train loop)."""

    def __init__(self, prompts: Sequence[Prompt], seed: int) -> None:
        self.prompts = list(prompts)
        self.rng = random.Random(seed)
        self._order: list[int] = []

    def _reshuffle(self) -> None:
        self._order = list(range(len(self.prompts)))
        self.rng.shuffle(self._order)

    def next_batch(self, n: int) -> list[Prompt]:
        out: list[Prompt] = []
        while len(out) < n:
            if not self._order:
                self._reshuffle()
            out.append(self.prompts[self._order.pop()])
        return out


def iter_batches(prompts: Sequence[Prompt], batch_size: int) -> Iterator[list[Prompt]]:
    """One-pass iterator (no shuffling); used for deterministic evaluation."""
    batch: list[Prompt] = []
    for p in prompts:
        batch.append(p)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_and_dump_splits(
    cfg: dict,
    out_dir: str | Path,
) -> PromptSplits:
    splits = load_prompt_pool(
        cfg["prompt_pool_path"],
        dims_whitelist=cfg["dims_whitelist"],
        train_size=cfg["train_split_size"],
        eval_size=cfg["eval_split_size"],
        fixed_size=cfg["fixed_eval_size"],
        seed=cfg["seed"],
    )
    out_dir = Path(out_dir)
    write_split(splits.train, out_dir / "evalverse_train.jsonl")
    write_split(splits.eval, out_dir / "evalverse_eval.jsonl")
    write_split(splits.fixed32, out_dir / "evalverse_fixed32.jsonl")
    print(
        f"[prompt_pool] wrote {len(splits.train)} train / "
        f"{len(splits.eval)} eval / {len(splits.fixed32)} fixed32 to {out_dir}"
    )
    return splits
