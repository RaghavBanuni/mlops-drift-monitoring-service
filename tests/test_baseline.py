"""The baseline is an artefact: it must survive a round trip and match itself exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.baseline import (
    OTHER_LABEL,
    Baseline,
    fit_baseline,
    infer_kinds,
)
from driftwatch.detect import psi_from_shares, scan_window
from driftwatch.metrics import bin_shares


class TestFitting:
    def test_a_thin_reference_is_refused(self):
        frame = pd.DataFrame({"x": np.arange(50.0)})
        with pytest.raises(ValueError, match="rows"):
            fit_baseline(frame, features=["x"], min_rows=200)

    def test_kinds_are_inferred_and_low_cardinality_numbers_are_categorical(self):
        frame = pd.DataFrame(
            {
                "amount": np.random.default_rng(0).normal(size=300),
                "tier": np.random.default_rng(1).integers(1, 4, 300),
                "region": ["a", "b"] * 150,
            }
        )
        kinds = infer_kinds(frame, ["amount", "tier", "region"])
        assert kinds == {"amount": "numeric", "tier": "categorical", "region": "categorical"}

    def test_declaring_a_column_twice_is_an_error(self):
        frame = pd.DataFrame({"x": np.random.default_rng(0).normal(size=300)})
        with pytest.raises(ValueError, match="both"):
            fit_baseline(frame, numeric=["x"], categorical=["x"])

    def test_reference_performance_is_recorded_when_outcomes_are_supplied(self, baseline):
        assert baseline.baseline_auc is not None
        assert 0.55 < baseline.baseline_auc < 1.0  # the synthetic model must actually rank
        assert 0.0 < baseline.positive_rate < 1.0
        assert baseline.prediction is not None

    def test_features_exclude_the_prediction_and_target(self, baseline):
        assert "prediction" not in baseline.feature_names
        assert "label" not in baseline.feature_names
        assert baseline.kind_of("income") == "numeric"
        assert baseline.kind_of("region") == "categorical"
        with pytest.raises(KeyError):
            baseline.kind_of("not_a_feature")


class TestStoredShares:
    """Why the reference bin masses are stored rather than assumed uniform."""

    def test_reference_scores_zero_against_itself(self, baseline, stable):
        report = scan_window(stable.reference, baseline, window=0)
        numeric = [item for item in report.features if item.kind == "numeric"]
        assert numeric
        for item in numeric:
            assert item.effect == pytest.approx(0.0, abs=1e-9)
        assert report.verdict == "stable"
        assert not report.flagged

    def test_tied_features_would_break_a_uniform_assumption(self):
        # 70% of the mass sits on one value, so the bins are genuinely unequal.  Assuming 1/k
        # per bin would report drift on the reference itself; the stored shares do not.
        values = np.concatenate([np.zeros(700), np.linspace(1, 10, 300)])
        frame = pd.DataFrame({"x": values})
        baseline = fit_baseline(frame, numeric=["x"], min_rows=200)
        profile = baseline.numeric["x"]

        assert profile.share_array().sum() == pytest.approx(1.0, abs=1e-6)
        assert profile.share_array().max() > 0.5  # unequal, as expected
        assert psi_from_shares(
            profile.share_array(), bin_shares(values, profile.edge_array())
        ) == pytest.approx(0.0, abs=1e-9)

        uniform = np.full(profile.n_bins, 1.0 / profile.n_bins)
        assert psi_from_shares(uniform, profile.share_array()) > 0.25  # the bug being avoided

    def test_reference_counts_sum_to_the_reference_size(self, baseline):
        profile = baseline.numeric["income"]
        assert profile.reference_counts().sum() == pytest.approx(profile.count, rel=1e-6)


class TestCategoricalProfile:
    def test_rare_levels_are_folded_into_one_bucket(self):
        rng = np.random.default_rng(2)
        common = rng.choice(["a", "b", "c"], size=900)
        tail = [f"id_{index}" for index in range(100)]  # 100 levels, one row each
        frame = pd.DataFrame({"key": np.concatenate([common, tail])})
        baseline = fit_baseline(frame, categorical=["key"], min_rows=200, min_category_share=0.01)
        profile = baseline.categorical["key"]

        assert OTHER_LABEL in profile.categories
        assert len(profile.categories) <= 5
        assert len(profile.rare) == 100
        assert sum(profile.shares) == pytest.approx(1.0, abs=1e-9)

    def test_folded_levels_are_not_reported_as_unseen(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {"key": np.concatenate([rng.choice(["a", "b"], size=800), [f"t{i}" for i in range(200)]])}
        )
        baseline = fit_baseline(frame, categorical=["key"], min_rows=200, min_category_share=0.02)
        profile = baseline.categorical["key"]

        # a rare level seen in the reference maps to "other", not to the unseen slot
        counts = profile.encode(["t5", "t6", "a"])
        assert counts[-1] == 0
        # a level the reference never saw does land there
        assert profile.encode(["brand_new"])[-1] == 1

    def test_unseen_slot_carries_no_reference_mass(self, baseline):
        profile = baseline.categorical["region"]
        assert profile.reference_counts()[-1] == 0.0
        assert profile.reference_counts().sum() == pytest.approx(profile.count, rel=1e-6)


class TestSerialisation:
    def test_round_trip_is_lossless(self, baseline, tmp_path):
        path = baseline.save(tmp_path / "nested" / "baseline.json")
        assert path.exists()
        reloaded = Baseline.load(path)

        assert reloaded.to_dict() == baseline.to_dict()
        assert reloaded.model_version == "test-1"
        original = baseline.numeric["income"]
        restored = reloaded.numeric["income"]
        assert restored.edge_array()[0] == -np.inf and restored.edge_array()[-1] == np.inf
        assert restored.share_array() == pytest.approx(original.share_array())
        assert restored.edges == original.edges

    def test_a_reloaded_baseline_still_scores_its_own_reference_as_stable(
        self, baseline, stable, tmp_path
    ):
        reloaded = Baseline.load(baseline.save(tmp_path / "baseline.json"))
        report = scan_window(stable.reference, reloaded, window=0)
        assert report.verdict == "stable"
        assert all(
            item.effect == pytest.approx(0.0, abs=1e-9)
            for item in report.features
            if item.kind == "numeric"
        )
