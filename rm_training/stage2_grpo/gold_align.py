"""rm_training.stage2_grpo.gold_align
==================================
Bipartite-matching coverage similarity between a student rubric and the
teacher (gold) rubric.

For each student rubric path we find its single best match in the gold
rubric (Hungarian one-to-one), then average the matched cosine
similarities. This implements the "rationale-RM-style soft recall"
described in the design: it scores **dimension coverage**, never the
scores themselves.

Inputs are plain ``list[str]``s of rubric path strings; the module
abstracts the embedder behind a tiny ``Embedder`` protocol so that the
GRPO trainer can inject either:
  • a real ``sentence-transformers`` (BGE) instance, or
  • a deterministic mock (for CPU dummy mode + unit tests).

Public API
----------
- ``cosine_matrix(A, B) -> np.ndarray``       ; A:(n,D), B:(m,D)
- ``hungarian_match(cost) -> (row_ind, col_ind)``
- ``bipartite_sim(student_paths, gold_paths, embedder) -> float``
- ``DummyEmbedder``                       ; deterministic, no torch
"""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _HAS_SCIPY = True
except ImportError:                          # pragma: no cover
    _HAS_SCIPY = False

__all__ = [
    "Embedder",
    "DummyEmbedder",
    "cosine_matrix",
    "hungarian_match",
    "bipartite_sim",
    "forward_chamfer_sim",
]


# ---------------------------------------------------------------------------
# Embedder protocol — anything with ``.encode(list[str]) -> np.ndarray``.
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    def encode(self, texts: Iterable[str]) -> np.ndarray: ...   # noqa: E704


class DummyEmbedder:
    """Hash-based deterministic 'embedder' for tests / CPU dummy mode.

    Produces unit-norm vectors whose pairwise cosine roughly tracks
    string overlap on whitespace tokens. NOT a real model — only good
    enough to make ``bipartite_sim(R, R) ≈ 1`` and
    ``bipartite_sim(R, unrelated) ≪ 1`` so unit tests can verify the
    matching logic itself.
    """
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec_for_token(self, tok: str) -> np.ndarray:
        # Stable per-token vector seeded from the token's hash.
        rng = np.random.default_rng(abs(hash(tok)) % (2**31))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(v) + 1e-8
        return v / n

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        out = []
        for t in texts:
            toks = (t or "").lower().split()
            if not toks:
                out.append(np.zeros(self.dim, dtype=np.float32))
                continue
            agg = np.mean([self._vec_for_token(tok) for tok in toks], axis=0)
            n = np.linalg.norm(agg) + 1e-8
            out.append((agg / n).astype(np.float32))
        return np.stack(out)


# ---------------------------------------------------------------------------
# Cosine and matching primitives.
# ---------------------------------------------------------------------------
def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Inputs may be unnormalised."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    a_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    b_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return (A @ B.T) / (a_norm * b_norm.T)


def hungarian_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve min-cost bipartite matching.

    Falls back to a greedy O(nm) match when scipy is missing — fine for
    small K (we expect K=5 for both student and gold).
    """
    cost = np.asarray(cost, dtype=np.float32)
    if _HAS_SCIPY:
        return _hungarian(cost)
    # ── Greedy fallback: pick the lowest-cost cell, mark its row+col,
    #    repeat. Returns the same (row_ind, col_ind) shape contract.
    n, m = cost.shape
    work = cost.copy()
    rows, cols = [], []
    for _ in range(min(n, m)):
        i, j = np.unravel_index(np.argmin(work), work.shape)
        rows.append(int(i))
        cols.append(int(j))
        work[i, :] = np.inf
        work[:, j] = np.inf
    return np.asarray(rows), np.asarray(cols)


# ---------------------------------------------------------------------------
# Public main: bipartite_sim().
# ---------------------------------------------------------------------------
def bipartite_sim(
    student_paths: list[str],
    gold_paths:    list[str],
    embedder:      Embedder | None = None,
) -> float:
    """Return one scalar in [-1, 1] capturing student↔gold rubric coverage.

    Implementation follows the design spec::

        sim(R̂, R*) = (1/n) Σ_j max_i  cos(d_i, d*_j)         #  (a)

    To realise the "max_i" with a *one-to-one* matching constraint we
    run Hungarian on the negated cosine matrix, taking the average of
    matched cosines. When |R̂| ≠ |R*|, we average over min(n, m) matches
    (Hungarian only matches min(n, m) pairs by definition).
    """
    if embedder is None:
        embedder = DummyEmbedder()
    if not student_paths or not gold_paths:
        return 0.0

    E_hat  = embedder.encode(student_paths)
    E_gold = embedder.encode(gold_paths)
    cos    = cosine_matrix(E_hat, E_gold)             # (n, m)
    cost   = -cos                                     # min-cost == max-sim
    row_i, col_j = hungarian_match(cost)
    matched = cos[row_i, col_j]
    if matched.size == 0:
        return 0.0
    return float(np.mean(matched))


# ---------------------------------------------------------------------------
# Public alt: forward_chamfer_sim().
# ---------------------------------------------------------------------------
def forward_chamfer_sim(
    student_paths: list[str],
    gold_paths:    list[str],
    embedder:      Embedder | None = None,
) -> float:
    """Forward Chamfer cosine: each student item finds its best gold neighbour.

    sim(R̂, R*) = (1 / |R̂|) · Σ_i  max_j  cos(d̂_i, d*_j)

    Why this differs from ``bipartite_sim``
    ---------------------------------------
    Hungarian 1:1 matching forces every student rubric to be paired with
    exactly one gold rubric and vice versa. That over-constrains the
    student in two ways:

    1. **Punishes novelty**: if the student writes a per-prompt-specific
       theme that has no analogue in the gold list, Hungarian still has
       to match it to *some* gold theme — usually a low-cosine pairing,
       which drags the average down.
    2. **Punishes selective coverage**: when |R*| > |R̂|, Hungarian leaves
       some gold rubrics unmatched. Because the average is over matched
       pairs, that's actually fine for the score — but its dual,
       1:1-with-Hungarian over |R̂| < |R*|, means the student is forced
       to cover a *fixed* gold subset chosen by the matching, regardless
       of which gold themes are most relevant for the current prompt.

    Forward Chamfer drops both constraints:
        * each student theme finds its best gold neighbour independently
          (no 1:1 budget — multiple students may map to the same gold);
        * gold themes the student didn't pick aren't counted (we don't
          penalise *under-coverage* — the rubric-PG already covers
          coverage via r_pref + r_format).

    Use it when the gold rubric is a "neighbourhood anchor" rather than
    "the unique correct answer" — exactly the warm-start anchor pattern.
    """
    if embedder is None:
        embedder = DummyEmbedder()
    if not student_paths or not gold_paths:
        return 0.0

    E_hat  = embedder.encode(student_paths)
    E_gold = embedder.encode(gold_paths)
    cos    = cosine_matrix(E_hat, E_gold)             # (n, m)
    if cos.size == 0:
        return 0.0
    # For each student row, take its max similarity over all gold cols,
    # then average across student rows.
    per_student_max = cos.max(axis=1)                 # (n,)
    return float(np.mean(per_student_max))
