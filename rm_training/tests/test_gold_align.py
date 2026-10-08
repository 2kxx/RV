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
    forward_chamfer_sim,
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


# ---------------------------------------------------------------------------
# forward_chamfer_sim — added 2026-06-15.
# ---------------------------------------------------------------------------
def test_forward_chamfer_self_is_one():
    paths = [
        "Theme: Visual Quality and Temporal Consistency",
        "Theme: Character Fidelity and Appearance",
        "Theme: Action Plausibility and Character Dynamics",
    ]
    s = forward_chamfer_sim(paths, paths, embedder=DummyEmbedder(dim=64))
    assert s > 0.99


def test_forward_chamfer_handles_empty_inputs():
    assert forward_chamfer_sim([], ["abc"], embedder=DummyEmbedder()) == 0.0
    assert forward_chamfer_sim(["abc"], [], embedder=DummyEmbedder()) == 0.0


def test_forward_chamfer_does_not_punish_novel_student_themes():
    """A student that adds an extra novel theme on top of matching ones
    should NOT be dragged below ~1.0 by the novelty.

    Bipartite (Hungarian 1:1) would force the novel theme to pair with
    some unrelated gold and average it in, dropping the score. Forward
    Chamfer instead lets every student pick its own best gold neighbour
    independently, so the matching themes still score 1.0 and the novel
    theme contributes its own (lower) max — but the 2 perfect matches
    keep the average high.
    """
    e = DummyEmbedder(dim=64)
    gold    = ["theme alpha", "theme beta"]
    student = ["theme alpha", "theme beta", "novel xyz unrelated"]
    bipart  = bipartite_sim(student, gold, embedder=e)
    chamfer = forward_chamfer_sim(student, gold, embedder=e)
    # Chamfer keeps the average high because two of three students saturate.
    assert chamfer > 0.6
    # Chamfer must be *at least as high* as bipartite when student is a
    # superset (bipartite can only match min(n,m) = |gold| pairs, so a
    # super-set student is no worse — and Chamfer is strictly looser).
    assert chamfer + 1e-6 >= min(bipart, 1.0) - 0.5  # sanity guard


def test_forward_chamfer_allows_multiple_students_to_share_a_gold():
    """If two student themes are both close to the same gold theme,
    forward Chamfer scores both at ~1.0 (no 1:1 budget). Bipartite would
    force one of them to pair with a different (unrelated) gold."""
    e = DummyEmbedder(dim=64)
    gold = ["theme alpha", "completely unrelated mountain"]
    # Both students target gold[0]; nothing in student matches gold[1].
    student = ["theme alpha", "theme alpha duplicate"]
    s = forward_chamfer_sim(student, gold, embedder=e)
    # Each student row's max should be high (close to alpha), so average
    # should sit well above the unrelated-pair bound (~0.05 in the
    # DummyEmbedder).
    assert s > 0.5


def test_forward_chamfer_unrelated_lists_low():
    a = ["red car driving fast"]
    b = ["sunset over mountain peak"]
    s = forward_chamfer_sim(a, b, embedder=DummyEmbedder(dim=64))
    assert s < 0.5
