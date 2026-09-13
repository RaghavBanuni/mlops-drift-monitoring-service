"""Distribution functions and multiple-testing correction, written from scratch.

A monitoring service that shells out to scipy for a p-value is fine; one that cannot explain
where the p-value came from is not.  Every function here is checked in the tests against
published critical values (chi-square 3.841 at 1 d.f. -> 0.05, Kolmogorov 1.3581 -> 0.05, and
so on), so the numbers are verifiable rather than trusted.

The series implementations follow the standard formulations in Numerical Recipes: the
alternating Kolmogorov series with a relative-term stopping rule, and the incomplete gamma
function split between its power series and its continued fraction at ``x = a + 1``, which is
where each branch converges quickly.
"""

from __future__ import annotations

import math

import numpy as np

_MAX_ITERATIONS = 200
_TINY = 1e-300


def kolmogorov_sf(x: float) -> float:
    """Survival function of the Kolmogorov distribution, ``Q(x) = P(K > x)``.

    Used for the asymptotic two-sample KS p-value.  The series
    ``Q(x) = 2 * sum_k (-1)^(k-1) exp(-2 k^2 x^2)`` converges fast for moderate ``x`` and is
    truncated on a relative-term rule; for very small ``x`` it degenerates, and 1.0 is the
    correct limit anyway.
    """
    if not math.isfinite(x):
        raise ValueError("x must be finite")
    if x <= 0.0:
        return 1.0
    scale = -2.0 * x * x
    factor = 2.0
    total = 0.0
    previous = 0.0
    for k in range(1, 101):
        term = factor * math.exp(scale * k * k)
        total += term
        if abs(term) <= 1e-3 * previous or abs(term) <= 1e-8 * abs(total):
            return float(min(max(total, 0.0), 1.0))
        factor = -factor
        previous = abs(term)
    return 1.0


def _lower_gamma_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(a, x)`` by its power series."""
    if x <= 0.0:
        return 0.0
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(_MAX_ITERATIONS):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_gamma_continued_fraction(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)`` by the modified Lentz algorithm."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b if b != 0.0 else 1.0 / _TINY
    h = d
    for index in range(1, _MAX_ITERATIONS + 1):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_sf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)``."""
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return float(min(max(1.0 - _lower_gamma_series(a, x), 0.0), 1.0))
    return float(min(max(_upper_gamma_continued_fraction(a, x), 0.0), 1.0))


def chi2_sf(statistic: float, dof: int) -> float:
    """Upper tail of the chi-square distribution."""
    if dof < 1:
        raise ValueError("dof must be at least 1")
    if statistic < 0:
        raise ValueError("the statistic must be non-negative")
    return gamma_sf(dof / 2.0, statistic / 2.0)


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal, via ``erfc``."""
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def benjamini_hochberg(
    pvalues: np.ndarray | list[float], alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.  Returns ``(rejected, qvalues)``.

    This is the correction a monitoring service actually needs.  Testing 40 features every day
    at alpha = 0.05 produces about two significant results per day from a perfectly stable
    pipeline - roughly 600 false alarms a year, which is how monitoring gets muted.  BH bounds
    the expected share of false discoveries *among the alerts raised* instead of the per-test
    error rate, and unlike Bonferroni it does not lose most of its power at 40 tests.

    Adjusted values are made monotone with a running minimum from the largest p-value down, so
    a q-value is never smaller than that of a more significant feature.
    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if values.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    if np.isnan(values).any():
        raise ValueError("pvalues must not contain NaN")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    count = values.size
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    ranks = np.arange(1, count + 1)

    qvalues_sorted = np.minimum.accumulate((ranked * count / ranks)[::-1])[::-1]
    qvalues_sorted = np.clip(qvalues_sorted, 0.0, 1.0)

    below = ranked <= alpha * ranks / count
    rejected_sorted = np.zeros(count, dtype=bool)
    if below.any():
        # every hypothesis up to the largest passing rank is rejected, including any whose own
        # p-value sits above its own threshold - that is the step-up part of the procedure
        rejected_sorted[: int(np.flatnonzero(below).max()) + 1] = True

    rejected = np.zeros(count, dtype=bool)
    qvalues = np.zeros(count, dtype=float)
    rejected[order] = rejected_sorted
    qvalues[order] = qvalues_sorted
    return rejected, qvalues


def bonferroni(pvalues: np.ndarray | list[float], alpha: float = 0.05) -> np.ndarray:
    """Bonferroni rejection mask, kept for the comparison in the evaluation harness."""
    values = np.asarray(pvalues, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    return values <= alpha / values.size


def bootstrap_ci(
    sample: np.ndarray,
    statistic,
    n_boot: int = 400,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic of one sample.

    Used for performance metrics, where the question is never "did AUC change" but "did it
    change by more than this window's sample size can explain".  A 900-row window gives an AUC
    standard error around 0.02, so a 0.01 drop is noise no matter how alarming the dashboard
    makes it look.
    """
    values = np.asarray(sample)
    if values.shape[0] < 2:
        raise ValueError("need at least two observations")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        picks = rng.integers(0, values.shape[0], values.shape[0])
        draws[index] = statistic(values[picks])
    tail = (1.0 - level) / 2.0
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(finite, tail)),
        float(np.quantile(finite, 1.0 - tail)),
    )
