"""Unit tests for ``rm_training.stage2_grpo.gold_align``.

Verifies that the bipartite-matching coverage similarity:
  • returns ~1.0 for identical rubric lists;
  • returns < 0.5 for clearly unrelated lists (with the dummy embedder);
  • is invariant to row/column permutations of the inputs.
"""

from __future__ import annotations

import numpy as np

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

from rm_training.stage2_grpo.gold_align import (
    DummyEmbedder,
    bipartite_sim,
    cosine_matrix,
    hungarian_match,
)


def test_dummy_embedder_unit_norm_when_nonempty():
    e = DummyEmbedder(dim=32)
    v = e.encode(["hello world"])[0]
    assert v.shape == (32,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


def test_dummy_embedder_zero_for_empty():
    e = DummyEmbedder(dim=16)
    v = e.encode([""])[0]
    assert np.linalg.norm(v) == 0.0


def test_cosine_matrix_shapes():
    A = np.random.RandomState(0).randn(3, 8)
    B = np.random.RandomState(1).randn(4, 8)
    M = cosine_matrix(A, B)
    assert M.shape == (3, 4)
    # Self-similarity must be ~1 along the diagonal.
    M2 = cosine_matrix(A, A)
    diag = np.diag(M2)
    assert np.allclose(diag, 1.0, atol=1e-5)


def test_hungarian_match_picks_diagonal_when_optimal():
    # Cost matrix where the diagonal is clearly cheapest.
    cost = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    rows, cols = hungarian_match(cost)
    pairs = sorted(zip(rows.tolist(), cols.tolist()))
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_bipartite_sim_self_is_one():
    paths = [
        "Theme: Visual Quality and Temporal Consistency",
        "Theme: Character Fidelity and Appearance",
        "Theme: Environment, Lighting, and Cinematic Style",
        "Theme: Action Plausibility and Character Dynamics",
        "Theme: Camera Motion, Framing, and Narrative Flow",
    ]
    s = bipartite_sim(paths, paths, embedder=DummyEmbedder(dim=64))
    assert s > 0.99


def test_bipartite_sim_invariant_to_permutation():
    paths = [
        "Theme: Visual Quality and Temporal Consistency",
        "Theme: Character Fidelity and Appearance",
        "Theme: Environment, Lighting, and Cinematic Style",
        "Theme: Action Plausibility and Character Dynamics",
        "Theme: Camera Motion, Framing, and Narrative Flow",
    ]
    perm = list(reversed(paths))
    e = DummyEmbedder(dim=64)
    s1 = bipartite_sim(paths, paths, embedder=e)
    s2 = bipartite_sim(paths, perm,  embedder=e)
    assert abs(s1 - s2) < 1e-5


def test_bipartite_sim_handles_empty_inputs():
    assert bipartite_sim([], ["abc"], embedder=DummyEmbedder()) == 0.0
    assert bipartite_sim(["abc"], [], embedder=DummyEmbedder()) == 0.0


def test_bipartite_sim_unrelated_lists_low():
    a = ["red car driving fast"]
    b = ["sunset over mountain peak"]
    s = bipartite_sim(a, b, embedder=DummyEmbedder(dim=64))
    # DummyEmbedder is hash-based so unrelated tokens must yield <0.5.
    assert s < 0.5
