"""The hand-rolled distributions are checked against published critical values.

The package deliberately ships no scipy, so these tests are what makes the p-values trustworthy:
if ``chi2_sf`` were wrong, every significance decision in the service would be wrong with it and
nothing else would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.stats import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    chi2_sf,
    gamma_sf,
    kolmogorov_sf,
    normal_sf,
)


class TestTailProbabilities:
    def test_normal_tail_matches_the_table(self):
        assert normal_sf(0.0) == pytest.approx(0.5, abs=1e-6)
        assert normal_sf(1.6448536) == pytest.approx(0.05, abs=1e-4)
        assert normal_sf(1.959964) == pytest.approx(0.025, abs=1e-4)
        assert normal_sf(2.5758293) == pytest.approx(0.005, abs=1e-4)

    def test_normal_tail_is_symmetric_and_monotone(self):
        assert normal_sf(-1.5) == pytest.approx(1.0 - normal_sf(1.5), abs=1e-9)
        grid = np.linspace(-4, 4, 50)
        values = [normal_sf(z) for z in grid]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))

    def test_chi_square_tail_matches_the_table(self):
        # 5% critical values for 1, 2, 5 and 9 degrees of freedom
        assert chi2_sf(3.8415, 1) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(5.9915, 2) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(11.0705, 5) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(16.9190, 9) == pytest.approx(0.05, abs=1e-3)
        # and a 1% value, to pin the shape rather than one point
        assert chi2_sf(21.6660, 9) == pytest.approx(0.01, abs=1e-3)

    def test_chi_square_edges(self):
        assert chi2_sf(0.0, 3) == pytest.approx(1.0)
        assert chi2_sf(1_000.0, 3) < 1e-12

    def test_chi_square_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            chi2_sf(1.0, 0)
        with pytest.raises(ValueError):
            chi2_sf(-1.0, 3)

    def test_gamma_tail_agrees_with_the_exponential_case(self):
        # Q(1, x) is exactly exp(-x): the series and the continued fraction must both land there
        for x in (0.05, 0.5, 2.0, 12.0, 40.0):
            assert gamma_sf(1.0, x) == pytest.approx(np.exp(-x), rel=1e-8, abs=1e-12)

    def test_kolmogorov_tail_matches_the_table(self):
        assert kolmogorov_sf(0.0) == pytest.approx(1.0)
        assert kolmogorov_sf(1.3581) == pytest.approx(0.05, abs=2e-3)
        assert kolmogorov_sf(1.6276) == pytest.approx(0.01, abs=2e-3)
        assert kolmogorov_sf(6.0) < 1e-9


class TestMultiplicityCorrection:
    def test_bh_qvalues_are_monotone_and_above_the_pvalues(self, rng):
        pvalues = np.sort(rng.uniform(size=40))
        _rejected, qvalues = benjamini_hochberg(pvalues, 0.05)
        assert np.all(qvalues >= pvalues - 1e-12)
        assert np.all(np.diff(qvalues) >= -1e-12)
        assert np.all(qvalues <= 1.0 + 1e-12)

    def test_bh_rejects_exactly_the_qvalues_below_alpha(self, rng):
        pvalues = rng.uniform(size=25) ** 3  # a mix of small and large
        rejected, qvalues = benjamini_hochberg(pvalues, 0.10)
        assert np.array_equal(rejected, qvalues <= 0.10)

    def test_bh_is_never_stricter_than_bonferroni(self, rng):
        pvalues = rng.uniform(size=60) ** 2
        bh, _ = benjamini_hochberg(pvalues, 0.05)
        assert int(bh.sum()) >= int(bonferroni(pvalues, 0.05).sum())

    def test_all_tiny_pvalues_are_rejected(self):
        rejected, qvalues = benjamini_hochberg([1e-6] * 5, 0.05)
        assert rejected.all()
        assert np.all(qvalues < 1e-5)

    def test_uniform_pvalues_stay_almost_entirely_unrejected(self):
        # the point of the correction: 200 null tests should not produce ten discoveries, which
        # is what an uncorrected alpha of 0.05 would hand an on-call engineer every single day
        pvalues = np.random.default_rng(3).uniform(size=200)
        rejected, _ = benjamini_hochberg(pvalues, 0.05)
        assert int(rejected.sum()) <= 2
        assert int(bonferroni(pvalues, 0.05).sum()) <= 1
        assert int((pvalues <= 0.05).sum()) >= 5  # uncorrected, for contrast

    def test_empty_input_is_handled(self):
        rejected, qvalues = benjamini_hochberg([], 0.05)
        assert rejected.size == 0 and qvalues.size == 0

    def test_input_validation(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([0.1, 1.4], 0.05)
        with pytest.raises(ValueError):
            benjamini_hochberg([0.1, float("nan")], 0.05)
        with pytest.raises(ValueError):
            benjamini_hochberg([0.1, 0.2], 1.5)


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        sample = np.random.default_rng(1).normal(5.0, 1.0, 400).reshape(-1, 1)
        low, high = bootstrap_ci(sample, lambda draw: float(draw[:, 0].mean()), n_boot=200, seed=2)
        assert low < sample[:, 0].mean() < high
        assert high - low < 0.5  # 400 rows of unit noise: the interval must be tight

    def test_the_interval_widens_as_the_sample_shrinks(self):
        rng = np.random.default_rng(4)
        statistic = lambda draw: float(draw[:, 0].mean())  # noqa: E731
        big = np.asarray(rng.normal(size=2_000)).reshape(-1, 1)
        small = big[:100]
        wide = bootstrap_ci(small, statistic, n_boot=200, seed=1)
        narrow = bootstrap_ci(big, statistic, n_boot=200, seed=1)
        assert (wide[1] - wide[0]) > 3 * (narrow[1] - narrow[0])

    def test_interval_is_reproducible(self):
        sample = np.random.default_rng(1).normal(size=200).reshape(-1, 1)
        statistic = lambda draw: float(draw[:, 0].mean())  # noqa: E731
        assert bootstrap_ci(sample, statistic, n_boot=100, seed=7) == bootstrap_ci(
            sample, statistic, n_boot=100, seed=7
        )

    def test_nan_statistics_do_not_poison_the_interval(self):
        sample = np.arange(50, dtype=float).reshape(-1, 1)
        low, high = bootstrap_ci(sample, lambda draw: float("nan"), n_boot=50, seed=0)
        assert np.isnan(low) and np.isnan(high)

    def test_bad_arguments_are_refused(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([[1.0]]), lambda draw: 0.0)
        with pytest.raises(ValueError):
            bootstrap_ci(np.arange(10.0).reshape(-1, 1), lambda draw: 0.0, level=1.5)
