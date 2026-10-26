"""Optimal one-to-one assignment (the Hungarian algorithm) for small score matrices.

Used to pair predicted line items with gold ones so that the pairing, not the order they were
printed or extracted in, decides what counts as correct.

Deliberate boundary: stdlib only and O(n^3), which is ample for the tens of rows on an invoice. It
exists instead of a scipy dependency because scipy ships no type information for strict checking
here; the brute-force cross-check in the tests is what makes a hand-written solver acceptable.
"""

from __future__ import annotations

from collections.abc import Sequence


def max_weight_assignment(scores: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Pair rows with columns to maximise the summed score.

    Args:
        scores: A rectangular matrix; ``scores[r][c]`` is the value of pairing row ``r`` with
            column ``c``. Rows and columns may differ in number.

    Returns:
        ``min(rows, cols)`` pairs ``(row, col)``, sorted by row, using each row and column at
        most once, with the largest possible total score.

    Raises:
        ValueError: If the rows have different lengths.
    """
    rows = len(scores)
    cols = len(scores[0]) if rows else 0
    if any(len(r) != cols for r in scores):
        raise ValueError("score matrix must be rectangular")
    if rows == 0 or cols == 0:
        return []
    transpose = rows > cols
    matrix = (
        [list(col) for col in zip(*scores, strict=True)] if transpose else [list(r) for r in scores]
    )
    pairs = _minimise([[-v for v in row] for row in matrix])
    if transpose:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def _minimise(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment of every row, for a matrix with no more rows than columns.

    The classic potentials formulation: rows are added one at a time and an augmenting path is
    found with Dijkstra-like relaxation over reduced costs. Index 0 is a sentinel.
    """
    n, m = len(cost), len(cost[0])
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    owner = [0] * (m + 1)  # owner[j]: row matched to column j (1-based), 0 if none
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        min_slack = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                reduced = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if reduced < min_slack[j]:
                    min_slack[j], way[j] = reduced, j0
                if min_slack[j] < delta:
                    delta, j1 = min_slack[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    min_slack[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    return [(owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j]]
