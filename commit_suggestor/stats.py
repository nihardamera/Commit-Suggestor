"""Uncertainty and agreement statistics for the v2 evaluation.

Every number the README reports comes with an interval from here:

- `stratified_bootstrap`: resamples examples *within* each test project, so a
  resample keeps the projects' sizes. The test set is 57% sphinx, so `macro`
  (the mean of per-project means) is reported alongside the pooled mean.
- `wilson`: interval for a proportion such as a human win rate. Better than
  the normal approximation when n is small or the rate is near 0 or 1.
- `cohen_kappa`: agreement beyond chance between two raters (me and the judge,
  or me and myself on a re-rating).
- `spearman` / `kendall_tau_b`: rank agreement, e.g. ROUGE-L vs my scores.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def scaled(self, factor: float, digits: int = 2) -> list[float]:
        """[estimate, low, high] multiplied (e.g. by 100 for percentages) and rounded."""
        return [round(factor * v, digits) for v in (self.estimate, self.low, self.high)]


def _strata_means(
    values: np.ndarray, groups: list[np.ndarray], picks: list[np.ndarray], macro: bool
) -> np.ndarray:
    sums = [values[g][p].sum(axis=1) for g, p in zip(groups, picks, strict=True)]
    if macro:
        return np.mean([s / len(g) for s, g in zip(sums, groups, strict=True)], axis=0)
    return np.sum(sums, axis=0) / len(values)


def stratified_bootstrap(
    values: Sequence[float],
    strata: Sequence[Hashable],
    *,
    macro: bool = False,
    n: int = 10_000,
    seed: int = 0,
) -> Interval:
    """95% percentile interval for the mean of `values`, resampling within strata.

    For a paired comparison pass the per-example differences (b - a).
    """
    if len(values) != len(strata) or not values:
        raise ValueError("values and strata must be non-empty and the same length")
    vals = np.asarray(values, dtype=float)
    labels = list(strata)
    groups = [
        np.array([i for i, s in enumerate(labels) if s == key]) for key in dict.fromkeys(labels)
    ]
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(g), size=(n, len(g))) for g in groups]
    means = _strata_means(vals, groups, picks, macro)
    identity = [np.arange(len(g))[None, :] for g in groups]
    point = float(_strata_means(vals, groups, identity, macro)[0])
    low, high = np.percentile(means, [2.5, 97.5])
    return Interval(point, float(low), float(high))


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a proportion."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else centre - half
    high = 1.0 if successes == n else centre + half  # exact at the ends, not 0.999...
    return Interval(p, low, high)


def confusion(
    a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable]
) -> list[list[int]]:
    """Rows are rater a's label, columns rater b's, in the order of `labels`."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b, strict=True):
        matrix[index[x]][index[y]] += 1
    return matrix


def kappa_from_confusion(matrix: Sequence[Sequence[int]]) -> float:
    """Cohen's kappa: (observed - chance agreement) / (1 - chance agreement)."""
    total = sum(map(sum, matrix))
    if total == 0:
        raise ValueError("empty confusion matrix")
    k = len(matrix)
    observed = sum(matrix[i][i] for i in range(k)) / total
    rows = [sum(matrix[i]) / total for i in range(k)]
    cols = [sum(matrix[i][j] for i in range(k)) / total for j in range(k)]
    chance = sum(r * c for r, c in zip(rows, cols, strict=True))
    return 1.0 if chance == 1 else (observed - chance) / (1 - chance)


def cohen_kappa(
    a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> float:
    labels = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    return kappa_from_confusion(confusion(a, b, labels))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties given their average rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation of the ranks."""
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b, which corrects for ties (common with 1-5 scores)."""
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    dx = np.sign(xs[:, None] - xs[None, :])
    dy = np.sign(ys[:, None] - ys[None, :])
    upper = np.triu_indices(len(xs), k=1)
    sx, sy = dx[upper], dy[upper]
    concordance = float((sx * sy).sum())
    # Pairs not tied in x, and not tied in y.
    untied_x, untied_y = float((sx != 0).sum()), float((sy != 0).sum())
    return concordance / sqrt(untied_x * untied_y)
