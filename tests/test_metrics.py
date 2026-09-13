"""Divergences and tests: identities first, then the properties the detector relies on."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.metrics import (
    NULL_LABEL,
    bin_counts,
    bin_shares,
    categorical_counts,
    chi_square_homogeneity,
    clean_numeric,
    cramers_v,
    jensen_shannon,
    ks_statistic,
    ks_test,
    null_rate,
    psi,
    psi_table,
    quantile_bin_edges,
    reference_spread,
    scaled_wasserstein,
    total_variation,
    unseen_share,
    wasserstein1,
)


@pytest.fixture
def reference():
    return np.random.default_rng(4).normal(size=3_000)


class TestCleaning:
    def test_non_finite_values_are_dropped(self):
        cleaned = clean_numeric([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0])
        assert cleaned.tolist() == [1.0, 2.0, 3.0]

    def test_null_rate_counts_nan_and_none(self):
        assert null_rate([1.0, np.nan, None, 4.0]) == pytest.approx(0.5)
        assert null_rate([]) == 0.0


class TestBinning:
    def test_edges_are_open_ended_and_monotone(self, reference):
        edges = quantile_bin_edges(reference, 10)
        assert edges[0] == -np.inf and edges[-1] == np.inf
        assert np.all(np.diff(edges) > 0)
        assert len(edges) == 11

    def test_edges_collapse_on_ties_instead_of_duplicating(self):
        # 70% zeros: ten quantiles cannot all be distinct, and duplicate edges would make the
        # bin widths zero and PSI infinite
        values = np.concatenate([np.zeros(700), np.linspace(1, 10, 300)])
        edges = quantile_bin_edges(values, 10)
        assert np.all(np.diff(edges) > 0)
        assert len(edges) < 11

    def test_counts_and_shares_agree(self, reference):
        edges = quantile_bin_edges(reference, 8)
        counts = bin_counts(reference, edges)
        shares = bin_shares(reference, edges)
        assert counts.sum() == reference.size
        assert shares.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(shares > 0)  # the epsilon floor is what keeps PSI finite

    def test_quantile_bins_are_roughly_equal_mass(self, reference):
        shares = bin_shares(reference, quantile_bin_edges(reference, 10))
        assert shares.max() < 0.15 and shares.min() > 0.05


class TestPSI:
    def test_psi_is_zero_against_itself(self, reference):
        assert psi(reference, reference, n_bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_psi_grows_with_the_shift(self, reference):
        small = psi(reference, reference + 0.25, n_bins=10)
        medium = psi(reference, reference + 0.75, n_bins=10)
        large = psi(reference, reference + 1.5, n_bins=10)
        assert 0 < small < medium < large

    def test_psi_table_decomposes_the_total(self, reference):
        current = reference * 1.3 + 0.4
        table = psi_table(reference, current, n_bins=10)
        assert table["contribution"].sum() == pytest.approx(psi(reference, current, n_bins=10), rel=1e-9)
        assert table["reference_share"].sum() == pytest.approx(1.0, abs=1e-6)
        # sorted worst-first so a human sees which part of the range moved
        assert table["contribution"].is_monotonic_decreasing

    def test_psi_is_symmetric(self, reference):
        current = reference + 0.6
        edges = quantile_bin_edges(reference, 10)
        forward = psi(reference, current, edges=edges)
        backward = psi(current, reference, edges=edges)
        assert forward == pytest.approx(backward, rel=1e-9)


class TestDistances:
    def test_ks_statistic_bounds(self, reference):
        assert ks_statistic(reference, reference) == pytest.approx(0.0)
        assert ks_statistic([0.0, 0.1, 0.2], [5.0, 5.1, 5.2]) == pytest.approx(1.0)

    def test_ks_test_separates_signal_from_noise(self, reference):
        other = np.random.default_rng(5).normal(size=3_000)
        _statistic, same = ks_test(reference, other)
        _statistic, shifted = ks_test(reference, other + 0.5)
        assert same > 0.01
        assert shifted < 1e-10

    def test_wasserstein_recovers_a_pure_translation(self):
        values = np.random.default_rng(6).uniform(size=5_000)
        assert wasserstein1(values, values + 2.0) == pytest.approx(2.0, abs=0.02)

    def test_scaled_wasserstein_divides_by_the_reference_spread(self, reference):
        current = reference + 1.0
        expected = wasserstein1(reference, current) / reference_spread(reference)
        assert scaled_wasserstein(reference, current) == pytest.approx(expected, rel=1e-9)
        assert reference_spread(reference) > 0

    def test_total_variation_bounds(self):
        p = np.array([0.5, 0.3, 0.2])
        assert total_variation(p, p) == pytest.approx(0.0)
        assert total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
        assert 0 <= total_variation(p, np.array([0.2, 0.3, 0.5])) <= 1

    def test_jensen_shannon_is_bounded_and_ordered(self):
        p = np.array([0.5, 0.5])
        near = jensen_shannon(p, np.array([0.45, 0.55]))
        far = jensen_shannon(p, np.array([0.95, 0.05]))
        disjoint = jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert 0 == pytest.approx(jensen_shannon(p, p), abs=1e-12)
        assert 0 < near < far < disjoint <= 1.0 + 1e-9
        assert disjoint >= 0.69  # log2 gives 1 bit, natural log 0.693: both are the maximum


class TestCategorical:
    def test_counts_reserve_a_slot_for_unseen_levels(self):
        counts = categorical_counts(["a", "b", "a", "zzz", None], ["a", "b"])
        assert counts.size == 3
        assert counts[0] == 2 and counts[1] == 1
        assert counts[2] == 2  # the unknown level and the null both land in the unseen slot
        assert unseen_share(counts) == pytest.approx(0.4)

    def test_nulls_are_a_category_when_the_reference_knew_them(self):
        counts = categorical_counts(["a", None, None], ["a", NULL_LABEL])
        assert counts.tolist() == [1, 2, 0]

    def test_chi_square_does_not_flag_identical_distributions(self):
        counts = np.array([300, 250, 200, 150, 100])
        statistic, p_value, dof, pooled = chi_square_homogeneity(counts, counts)
        assert statistic == pytest.approx(0.0, abs=1e-9)
        assert p_value == pytest.approx(1.0)
        assert dof == 4 and pooled == 0

    def test_chi_square_flags_a_real_reallocation(self):
        reference = np.array([500.0, 300.0, 200.0])
        current = np.array([200, 300, 500])
        _statistic, p_value, _dof, _pooled = chi_square_homogeneity(reference, current)
        assert p_value < 1e-10

    def test_thin_categories_are_pooled_rather_than_trusted(self):
        # expected counts below five make the chi-square approximation unreliable; pooling is
        # the standard fix and it must show up in the returned dof
        reference = np.array([100.0, 100.0, 1.0, 1.0, 1.0])
        current = np.array([90, 110, 0, 1, 2])
        _statistic, _p, dof, pooled = chi_square_homogeneity(reference, current, min_expected=5.0)
        assert pooled >= 2
        assert dof < 4

    def test_cramers_v_is_a_bounded_effect_size(self):
        assert cramers_v(0.0, 500, 3) == pytest.approx(0.0)
        assert 0 <= cramers_v(120.0, 500, 3) <= 1
        assert cramers_v(50.0, 1_000, 2) < cramers_v(500.0, 1_000, 2)
