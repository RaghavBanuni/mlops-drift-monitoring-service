"""The detection rule: adaptive thresholds, FDR correction, and refusing to guess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.baseline import fit_baseline
from driftwatch.detect import (
    DriftPolicy,
    binned_wasserstein,
    decision_table,
    psi_from_shares,
    psi_noise_floor,
    scan_window,
    tvd_noise_floor,
)


class TestNoiseFloor:
    def test_the_formula_is_exact(self):
        assert psi_noise_floor(1_000, 500, 10) == pytest.approx(9 * (1 / 1_000 + 1 / 500))
        assert psi_noise_floor(10_000, 10_000, 20) == pytest.approx(19 * 2 / 10_000)

    def test_the_floor_falls_as_windows_grow(self):
        assert psi_noise_floor(1_000, 200, 10) > psi_noise_floor(1_000, 2_000, 10)

    def test_the_floor_rises_with_more_bins(self):
        assert psi_noise_floor(1_000, 1_000, 20) > psi_noise_floor(1_000, 1_000, 5)

    def test_bad_arguments_are_refused(self):
        with pytest.raises(ValueError):
            psi_noise_floor(0, 100, 10)
        with pytest.raises(ValueError):
            psi_noise_floor(100, 100, 1)

    def test_the_floor_predicts_what_undrifted_data_actually_does(self):
        """The claim behind the adaptive threshold, measured rather than asserted."""
        rng = np.random.default_rng(12)
        reference = pd.DataFrame({"x": rng.normal(size=4_000)})
        baseline = fit_baseline(reference, numeric=["x"], n_bins=10)
        floor = psi_noise_floor(4_000, 300, 10)

        observed = []
        for _ in range(20):
            window = pd.DataFrame({"x": rng.normal(size=300)})
            report = scan_window(window, baseline)
            observed.append(report.features[0].effect)

        median = float(np.median(observed))
        assert 0.2 * floor < median < 3.0 * floor  # the estimate is the right order of magnitude
        assert max(observed) < 3.0 * floor + 0.1  # and nothing crosses the adaptive threshold

    def test_categorical_floor_behaves_like_the_numeric_one(self):
        shares = np.array([0.4, 0.3, 0.2, 0.1])
        assert tvd_noise_floor(shares, 1_000, 200) > tvd_noise_floor(shares, 1_000, 2_000)
        assert tvd_noise_floor(shares, 10_000, 10_000) < 0.02
        with pytest.raises(ValueError):
            tvd_noise_floor(shares, 0, 100)


class TestPrimitives:
    def test_psi_from_shares_needs_positive_mass(self):
        with pytest.raises(ValueError):
            psi_from_shares([0.5, 0.5, 0.0], [0.4, 0.4, 0.2])
        with pytest.raises(ValueError):
            psi_from_shares([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_psi_from_shares_is_zero_on_identical_input(self):
        shares = np.array([0.2, 0.5, 0.3])
        assert psi_from_shares(shares, shares) == pytest.approx(0.0, abs=1e-12)

    def test_binned_wasserstein_measures_a_translation(self):
        points = np.array([0.0, 1.0, 2.0, 3.0])
        left = np.array([0.97, 0.01, 0.01, 0.01])
        right = np.array([0.01, 0.01, 0.01, 0.97])
        assert binned_wasserstein(left, right, points) > 2.5
        assert binned_wasserstein(left, left, points) == pytest.approx(0.0, abs=1e-12)


class TestPolicy:
    def test_incoherent_thresholds_are_refused(self):
        with pytest.raises(ValueError):
            DriftPolicy(psi_warn=0.3, psi_alert=0.2).validate()
        with pytest.raises(ValueError):
            DriftPolicy(alpha=0.0).validate()
        with pytest.raises(ValueError):
            DriftPolicy(noise_multiple=0.5).validate()
        with pytest.raises(ValueError):
            DriftPolicy(min_effective_rows=5).validate()


class TestScanning:
    def test_a_stable_window_is_stable(self, baseline, stable):
        report = scan_window(stable.windows[1], baseline, window=1)
        assert report.verdict == "stable"
        assert not report.flagged
        assert all(item.severity == "none" for item in report.features)
        assert all(item.q_value is not None for item in report.features)

    def test_a_real_shift_is_flagged_on_the_right_features(self, baseline, shifted):
        report = scan_window(shifted.windows[6], baseline, window=6)
        flagged = {item.feature for item in report.flagged}

        assert "income" in flagged  # shifted by 0.8 sd at the change point
        assert "region" not in flagged  # untouched: no collateral alarms
        assert report.verdict in ("investigate", "action_required")

        income = next(item for item in report.features if item.feature == "income")
        assert income.effect > income.noise_floor * 3
        assert income.q_value < 0.05
        assert income.detail["mean_shift_in_iqr"] > 0.2

    def test_the_prediction_column_is_monitored_too(self, baseline, shifted):
        report = scan_window(
            shifted.windows[6], baseline, window=6, prediction_column="prediction"
        )
        prediction = [item for item in report.features if item.kind == "prediction"]
        assert len(prediction) == 1
        assert prediction[0].feature == "prediction"

    def test_a_blocked_feature_is_skipped_not_scored(self, baseline, stable):
        window = stable.windows[2].copy()
        window["income"] = "n/a"
        report = scan_window(window, baseline, window=2)

        income = next(item for item in report.features if item.feature == "income")
        assert income.severity == "skipped"
        assert np.isnan(income.effect)
        assert "schema" in income.note
        # the rest of the window is still graded
        assert any(item.severity == "none" for item in report.features)
        assert report.verdict == "action_required"  # a broken column is not a quiet event

    def test_an_empty_window_skips_everything_without_raising(self, baseline, stable):
        report = scan_window(stable.windows[0].iloc[0:0], baseline, window=9)
        assert report.rows == 0
        assert all(item.severity == "skipped" for item in report.features)
        assert len(report.features) == len(baseline.feature_names)
        # the all-NaN case that a naive sort key would crash on
        frame = report.frame()
        assert len(frame) == len(baseline.feature_names)
        assert frame["effect"].isna().all()

    def test_too_few_rows_is_reported_as_such_rather_than_guessed(self, baseline, stable):
        report = scan_window(stable.windows[0].head(30), baseline, window=0)
        assert all(item.severity == "insufficient_data" for item in report.features)
        assert all(item.p_value is None for item in report.features)
        assert not report.flagged
        assert len(report.frame()) == len(baseline.feature_names)

    def test_significance_only_ever_demotes(self, baseline, shifted):
        window = shifted.windows[5]
        strict = scan_window(window, baseline, DriftPolicy(require_significance=True), window=5)
        loose = scan_window(window, baseline, DriftPolicy(require_significance=False), window=5)

        strict_flags = {item.feature for item in strict.flagged}
        loose_flags = {item.feature for item in loose.flagged}
        assert strict_flags <= loose_flags

    def test_reports_serialise_completely(self, baseline, stable):
        payload = scan_window(stable.windows[3], baseline, window=3).to_dict()
        assert payload["window"] == 3
        assert payload["verdict"] in ("stable", "investigate", "action_required")
        assert len(payload["features"]) == len(baseline.feature_names)
        assert "schema" in payload
        for feature in payload["features"]:
            assert set(feature) >= {"feature", "effect", "severity", "note", "q_value"}

    def test_frame_puts_the_worst_first(self, baseline, shifted):
        frame = scan_window(shifted.windows[7], baseline, window=7).frame()
        ranks = {"alert": 0, "warn": 1, "insufficient_data": 2, "skipped": 3, "none": 4}
        order = [ranks[value] for value in frame["severity"]]
        assert order == sorted(order)

    def test_a_non_frame_is_refused(self, baseline):
        with pytest.raises(TypeError):
            scan_window({"income": [1, 2, 3]}, baseline)


class TestDecisionTable:
    def test_all_four_cases_are_documented(self):
        table = decision_table()
        assert len(table) == 4
        assert list(table.columns) == ["feature drift", "quality drop", "reading", "action"]
        assert table["reading"].str.contains("concept drift").any()
