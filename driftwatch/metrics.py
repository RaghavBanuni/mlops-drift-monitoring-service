"""Drift metrics: effect sizes first, p-values second.

The ordering matters more than the formulas.  With 50,000 rows a week, a Kolmogorov-Smirnov
test will call a 0.2% shift in the mean "significant" - correctly, and uselessly.  p-values
answer *is this shift larger than sampling noise*, which at production volumes is almost always
yes.  The question a retraining decision needs is *is this shift large enough to matter*, and
that is an effect size: PSI, total variation, Wasserstein distance scaled by the reference
spread.  So every detector here reports both, and the alerting rule in :mod:`driftwatch.detect`
gates on the effect size, using the p-value only to suppress small-sample noise.

Two implementation details that are ordinary bugs elsewhere:

* **Bin edges come from the reference and never move.**  Re-deriving quantile edges from the
  current window is the classic PSI bug: the bins follow the drift, each bin keeps roughly its
  reference share, and PSI stays near zero while the distribution walks away.
* **Missing values are a signal, not a nuisance.**  Numeric tests drop nulls (they have no
  place on the real line), so the null rate is measured and reported separately.  Dropping
  nulls silently is how a pipeline that started emitting 30% nulls passes every drift check.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .stats import chi2_sf, kolmogorov_sf

NULL_LABEL = "__missing__"
UNSEEN_LABEL = "__unseen__"

# The 0.1 / 0.25 PSI thresholds are a credit-scoring convention, not a theorem: they were
# chosen for scorecard monitoring on tens of thousands of accounts and they travel badly to
# other bin counts and sample sizes. They are the default here because a shared convention
# beats an arbitrary one, and they are configurable everywhere they are used.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25


def clean_numeric(values) -> np.ndarray:
    """Finite float view of a numeric column.  Nulls and infinities are removed."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def null_rate(values) -> float:
    """Share of values that are null or non-finite - tracked in its own right."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        return 0.0
    return float(1.0 - np.isfinite(array).mean())


def quantile_bin_edges(reference, n_bins: int = 10) -> np.ndarray:
    """Reference quantile edges with open tails, deduplicated.

    Equal-frequency bins are used rather than equal-width ones so that a skewed feature does
    not end up with nine empty bins and one full one.  Ties collapse edges, which legitimately
    reduces the bin count: a feature that is 80% zeros cannot support ten distinct quantiles,
    and pretending otherwise inflates PSI with empty bins.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    clean = clean_numeric(reference)
    if clean.size < 2:
        raise ValueError("the reference sample needs at least two finite values")
    inner = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def bin_counts(values, edges: np.ndarray) -> np.ndarray:
    """Counts per bin, where ``edges`` are the fixed reference edges."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 3:
        raise ValueError("edges must be one-dimensional with at least three entries")
    clean = clean_numeric(values)
    index = np.searchsorted(edges[1:-1], clean, side="right")
    return np.bincount(index, minlength=edges.size - 1).astype(np.int64)


def bin_shares(values, edges: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Bin shares, floored at ``epsilon`` so an empty bin cannot make PSI infinite."""
    counts = bin_counts(values, edges)
    total = counts.sum()
    if total == 0:
        raise ValueError("no finite values to bin")
    shares = np.clip(counts / total, epsilon, None)
    return shares / shares.sum()


def psi(
    reference,
    current,
    edges: np.ndarray | None = None,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index, ``sum (c - r) * ln(c / r)`` over reference bins.

    PSI is the symmetrised KL divergence (Jeffreys divergence) of the binned distributions, so
    it is symmetric in its arguments - a useful property to test, and a reminder that it says
    "these differ by this much", not "this got worse".
    """
    binning = quantile_bin_edges(reference, n_bins) if edges is None else np.asarray(edges, float)
    reference_shares = bin_shares(reference, binning, epsilon)
    current_shares = bin_shares(current, binning, epsilon)
    return float(np.sum((current_shares - reference_shares) * np.log(current_shares / reference_shares)))


def psi_table(
    reference,
    current,
    edges: np.ndarray | None = None,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> pd.DataFrame:
    """Per-bin PSI contributions - which part of the distribution moved, and where to.

    A single PSI number tells you something changed; this table is what makes it actionable,
    because "the top decile emptied out" and "the whole distribution shifted right" have
    different causes and different fixes.
    """
    binning = quantile_bin_edges(reference, n_bins) if edges is None else np.asarray(edges, float)
    reference_shares = bin_shares(reference, binning, epsilon)
    current_shares = bin_shares(current, binning, epsilon)
    contribution = (current_shares - reference_shares) * np.log(current_shares / reference_shares)
    return pd.DataFrame(
        {
            "bin": [
                f"({binning[index]:.4g}, {binning[index + 1]:.4g}]"
                for index in range(binning.size - 1)
            ],
            "reference_share": np.round(reference_shares, 5),
            "current_share": np.round(current_shares, 5),
            "contribution": np.round(contribution, 5),
        }
    ).sort_values("contribution", ascending=False, ignore_index=True)


def ks_statistic(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two ECDFs."""
    a = np.sort(clean_numeric(reference))
    b = np.sort(clean_numeric(current))
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples need at least one finite value")
    merged = np.union1d(a, b)
    cdf_a = np.searchsorted(a, merged, side="right") / a.size
    cdf_b = np.searchsorted(b, merged, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_test(reference, current) -> tuple[float, float]:
    """``(statistic, p_value)`` for the two-sample KS test.

    The p-value uses the asymptotic Kolmogorov distribution with the standard small-sample
    correction ``(sqrt(n_e) + 0.12 + 0.11 / sqrt(n_e)) * D``.  It is an approximation, and it
    is deliberately *not* the alerting criterion: at 50,000 rows this p-value is essentially a
    function of sample size, which is why the effect size gates the alert.
    """
    statistic = ks_statistic(reference, current)
    n1 = clean_numeric(reference).size
    n2 = clean_numeric(current).size
    effective = math.sqrt(n1 * n2 / (n1 + n2))
    return statistic, kolmogorov_sf((effective + 0.12 + 0.11 / effective) * statistic)


def wasserstein1(reference, current) -> float:
    """Exact 1-Wasserstein (earth mover) distance between two empirical distributions.

    Computed as the integral of ``|F_ref - F_cur|`` over the merged support.  Unlike KS it
    accumulates the whole displacement rather than the single largest gap, and it is in the
    units of the feature - which is why it is scaled by the reference spread before use.
    """
    a = np.sort(clean_numeric(reference))
    b = np.sort(clean_numeric(current))
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples need at least one finite value")
    support = np.union1d(a, b)
    if support.size == 1:
        return 0.0
    cdf_a = np.searchsorted(a, support[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, support[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(support)))


def reference_spread(reference) -> float:
    """Robust scale of the reference: IQR, falling back to standard deviation, then 1.0."""
    clean = clean_numeric(reference)
    if clean.size == 0:
        return 1.0
    iqr = float(np.quantile(clean, 0.75) - np.quantile(clean, 0.25))
    if iqr > 1e-12:
        return iqr
    deviation = float(clean.std())
    return deviation if deviation > 1e-12 else 1.0


def scaled_wasserstein(reference, current) -> float:
    """Wasserstein distance in units of the reference IQR, so it compares across features.

    0.1 means "the distribution moved by a tenth of its own interquartile range", which is a
    sentence a product owner can act on. A raw distance of 4.2 is not.
    """
    return float(wasserstein1(reference, current) / reference_spread(reference))


def jensen_shannon(reference_shares: np.ndarray, current_shares: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, bounded in ``[0, 1]``."""
    p = np.asarray(reference_shares, dtype=float)
    q = np.asarray(current_shares, dtype=float)
    if p.shape != q.shape:
        raise ValueError("the two distributions must have the same number of bins")
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("distributions must have positive mass")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def entropy(values: np.ndarray) -> float:
        positive = values[values > 0]
        return float(-np.sum(positive * np.log2(positive)))

    return float(max(entropy(m) - 0.5 * (entropy(p) + entropy(q)), 0.0))


def total_variation(reference_shares: np.ndarray, current_shares: np.ndarray) -> float:
    """Total variation distance: the share of probability mass that moved."""
    p = np.asarray(reference_shares, dtype=float)
    q = np.asarray(current_shares, dtype=float)
    if p.shape != q.shape:
        raise ValueError("the two distributions must have the same number of bins")
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("distributions must have positive mass")
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def categorical_counts(values, categories: list[str]) -> np.ndarray:
    """Counts over the reference categories, with one extra slot for unseen levels.

    Nulls map to their own category rather than being dropped, and anything the reference never
    saw lands in :data:`UNSEEN_LABEL`.  New levels are the most common categorical incident in
    production - a vendor renames ``GB`` to ``UK`` - and they are usually invisible to a test
    that silently drops unknown keys.
    """
    if len(categories) == 0:
        raise ValueError("categories must not be empty")
    lookup = {category: index for index, category in enumerate(categories)}
    series = pd.Series(list(values), dtype="object")
    keys = series.where(series.notna(), NULL_LABEL).astype(str)
    index = keys.map(lookup).fillna(len(categories)).astype(int).to_numpy()
    return np.bincount(index, minlength=len(categories) + 1).astype(np.int64)


def unseen_share(counts: np.ndarray) -> float:
    """Share of the window that fell into the unseen-category slot (the last one)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    return float(counts[-1] / total) if total > 0 else 0.0


def chi_square_homogeneity(
    reference_counts: np.ndarray, current_counts: np.ndarray, min_expected: float = 5.0
) -> tuple[float, float, int, int]:
    """Chi-square test of homogeneity on a 2 x k table.

    Returns ``(statistic, p_value, dof, pooled_categories)``.

    Rare levels are pooled until every expected count clears ``min_expected``, because the
    chi-square approximation breaks down on thin cells and a long categorical tail otherwise
    manufactures significance out of single observations.  ``pooled_categories`` is returned so
    the report can say how much of the tail was folded together rather than hiding it.
    """
    reference = np.asarray(reference_counts, dtype=float)
    current = np.asarray(current_counts, dtype=float)
    if reference.shape != current.shape:
        raise ValueError("both count vectors must have the same length")
    if reference.sum() <= 0 or current.sum() <= 0:
        raise ValueError("both samples must be non-empty")

    column_total = reference + current
    grand = reference.sum() + current.sum()
    smaller = min(reference.sum(), current.sum())
    keep = (smaller * column_total / grand) >= min_expected
    pooled = int((~keep).sum())

    if keep.sum() < 1:
        return 0.0, 1.0, 0, pooled
    reference_effective = np.append(reference[keep], reference[~keep].sum())
    current_effective = np.append(current[keep], current[~keep].sum())
    if pooled == 0:  # nothing was folded together, so drop the empty pooled column
        reference_effective = reference_effective[:-1]
        current_effective = current_effective[:-1]

    rows = np.vstack([reference_effective, current_effective])
    row_totals = rows.sum(axis=1, keepdims=True)
    column_totals = rows.sum(axis=0, keepdims=True)
    total = rows.sum()
    if rows.shape[1] < 2 or total <= 0:
        return 0.0, 1.0, 0, pooled

    expected = row_totals @ column_totals / total
    usable = expected > 0
    statistic = float((((rows - expected) ** 2 / np.where(usable, expected, 1.0)) * usable).sum())
    dof = int(rows.shape[1] - 1)
    return statistic, float(chi2_sf(statistic, dof)), dof, pooled


def cramers_v(statistic: float, n: int, dof: int) -> float:
    """Cramer's V for a 2 x k table: the chi-square statistic turned into an effect size."""
    if n <= 0:
        raise ValueError("n must be positive")
    if dof < 1:
        return 0.0
    return float(min(math.sqrt(max(statistic, 0.0) / n), 1.0))
