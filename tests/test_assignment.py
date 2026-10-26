"""Tests for the assignment solver, cross-checked against brute force."""

import itertools
import random

import pytest

from docproof.assignment import max_weight_assignment


def brute_force_best(scores: list[list[float]]) -> float:
    rows, cols = len(scores), len(scores[0])
    if rows <= cols:
        return max(
            sum(scores[r][c] for r, c in enumerate(perm))
            for perm in itertools.permutations(range(cols), rows)
        )
    return max(
        sum(scores[r][c] for c, r in enumerate(perm))
        for perm in itertools.permutations(range(rows), cols)
    )


@pytest.mark.parametrize("seed", range(60))
def test_matches_brute_force_on_random_rectangular_matrices(seed: int) -> None:
    rng = random.Random(seed)
    rows, cols = rng.randint(1, 6), rng.randint(1, 6)
    scores = [[float(rng.randint(0, 4)) for _ in range(cols)] for _ in range(rows)]
    pairs = max_weight_assignment(scores)
    assert len(pairs) == min(rows, cols)
    assert len({r for r, _ in pairs}) == len(pairs)
    assert len({c for _, c in pairs}) == len(pairs)
    assert sum(scores[r][c] for r, c in pairs) == brute_force_best(scores)


def test_prefers_the_global_optimum_over_greedy() -> None:
    # Greedy would take (0, 0) for 3 and then be stuck with 0; the optimum is 2 + 2.
    assert max_weight_assignment([[3.0, 2.0], [2.0, 0.0]]) == [(0, 1), (1, 0)]


def test_empty_and_ragged_inputs() -> None:
    assert max_weight_assignment([]) == []
    assert max_weight_assignment([[]]) == []
    with pytest.raises(ValueError, match="rectangular"):
        max_weight_assignment([[1.0, 2.0], [1.0]])
