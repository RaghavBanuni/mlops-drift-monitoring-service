"""Quality measurement under late labels, where most monitoring quietly goes wrong."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.performance import (
    LabelBuffer,
    accuracy_at,
    brier_score,
    check_performance,
    coverage_report,
    positive_rate_shift,
    roc_auc,
    segment_coverage,
)


class TestPointMetrics:
    def test_auc_of_a_perfect_and_a_reversed_ranking(self):
        labels = [0, 0, 1, 1]
        assert roc_auc(labels, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
        assert roc_auc(labels, [0.4, 0.3, 0.2, 0.1]) == pytest.approx(0.0)

    def test_tied_scores_give_exactly_one_half(self):
        # the midrank correction: a model emitting one constant score has no ranking ability
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)

    def test_auc_matches_a_hand_computed_case(self):
        # two positives, three negatives; count the concordant pairs by hand: 5 of 6
        labels = [1, 0, 1, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        assert roc_auc(labels, scores) == pytest.approx(5 / 6)

    def test_a_single_class_window_is_undefined_not_zero(self):
        assert np.isnan(roc_auc([1, 1, 1], [0.2, 0.3, 0.4]))
        assert np.isnan(roc_auc([], []))

    def test_bad_input_is_refused(self):
        with pytest.raises(ValueError):
            roc_auc([0, 1], [0.5])
        with pytest.raises(ValueError):
            roc_auc([0, 2], [0.5, 0.6])

    def test_brier_and_accuracy(self):
        assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
        assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)
        assert accuracy_at([1, 0, 1], [0.9, 0.1, 0.2], threshold=0.5) == pytest.approx(2 / 3)


class TestLabelBuffer:
    def test_labels_join_by_id_and_record_their_lag(self):
        buffer = LabelBuffer()
        buffer.add_predictions(["a", "b", "c"], [0.1, 0.5, 0.9], window=0)
        assert buffer.add_labels(["a", "c"], [0, 1], window=3) == 2

        frame = buffer.frame()
        assert len(frame) == 2
        assert buffer.unmatched == 1  # b has not resolved yet
        assert set(frame["lag"]) == {3}

    def test_an_unknown_id_is_counted_rather_than_silently_dropped(self):
        buffer = LabelBuffer()
        buffer.add_predictions(["a"], [0.4], window=0)
        assert buffer.add_labels(["ghost"], [1], window=1) == 0
        assert buffer.late_labels == 1

    def test_the_buffer_is_bounded(self):
        buffer = LabelBuffer(maxlen=5)
        buffer.add_predictions([str(index) for index in range(20)], np.linspace(0, 1, 20), window=0)
        assert buffer.unmatched == 5
        assert buffer.add_labels(["0"], [1], window=1) == 0  # evicted, so counted as too late
        assert buffer.late_labels == 1

    def test_mismatched_lengths_are_refused(self):
        buffer = LabelBuffer()
        with pytest.raises(ValueError):
            buffer.add_predictions(["a", "b"], [0.1], window=0)
        with pytest.raises(ValueError):
            buffer.add_labels(["a"], [0, 1], window=0)

    def test_coverage_is_reported_with_the_metric(self):
        buffer = LabelBuffer()
        buffer.add_predictions([str(index) for index in range(100)], np.linspace(0, 1, 100), 0)
        buffer.add_labels([str(index) for index in range(30)], np.zeros(30), 2)

        coverage = coverage_report(buffer)
        assert coverage["labelled"] == 30
        assert coverage["awaiting_label"] == 70
        assert coverage["coverage"] == pytest.approx(0.3)
        assert coverage["median_lag_windows"] == 2.0

    def test_coverage_on_an_empty_buffer_is_zero_not_an_error(self):
        coverage = coverage_report(LabelBuffer())
        assert coverage["coverage"] == 0.0
        assert np.isnan(coverage["median_lag_windows"])

    def test_segment_coverage_exposes_differential_label_delay(self):
        buffer = LabelBuffer()
        buffer.add_predictions(
            ["a", "b", "c", "d"], [0.1, 0.2, 0.3, 0.4], window=0, segments=["fast", "fast", "slow", "slow"]
        )
        buffer.add_labels(["a", "b"], [1, 0], window=1)
        buffer.add_labels(["c"], [1], window=9)

        frame = segment_coverage(buffer)
        assert set(frame["segment"]) == {"fast", "slow"}
        fast = frame[frame["segment"] == "fast"].iloc[0]
        slow = frame[frame["segment"] == "slow"].iloc[0]
        assert fast["mean_lag"] < slow["mean_lag"]
        assert frame["share_of_labelled"].sum() == pytest.approx(1.0)


class TestQualityCheck:
    def test_too_few_labels_is_its_own_verdict(self):
        rng = np.random.default_rng(0)
        check = check_performance(
            rng.integers(0, 2, 50), rng.random(50), baseline_auc=0.8, baseline_positive_rate=0.4
        )
        assert check.verdict == "insufficient_labels"
        assert np.isnan(check.auc)
        assert check.to_dict()["auc"] is None

    def test_a_single_class_window_cannot_be_judged(self):
        check = check_performance(
            np.ones(400), np.random.default_rng(1).random(400), baseline_auc=0.8, baseline_positive_rate=0.4
        )
        assert check.verdict == "insufficient_labels"

    def test_a_collapse_is_called_degraded(self):
        rng = np.random.default_rng(2)
        labels = rng.integers(0, 2, 800)
        scores = rng.random(800)  # no relationship at all: AUC near 0.5
        check = check_performance(labels, scores, baseline_auc=0.85, baseline_positive_rate=0.5)
        assert check.verdict == "degraded"
        assert check.auc_high < 0.85
        assert "below the baseline" in check.note

    def test_noise_around_the_baseline_is_called_stable(self):
        rng = np.random.default_rng(3)
        labels = (rng.random(800) < 0.4).astype(int)
        scores = labels * 0.4 + rng.random(800) * 0.6
        measured = roc_auc(labels, scores)
        check = check_performance(labels, scores, baseline_auc=measured, baseline_positive_rate=0.4)
        assert check.verdict == "stable"
        assert check.auc_low <= measured <= check.auc_high

    def test_an_improvement_is_not_reported_as_decay(self):
        rng = np.random.default_rng(4)
        labels = (rng.random(800) < 0.5).astype(int)
        scores = labels * 0.8 + rng.random(800) * 0.2
        check = check_performance(labels, scores, baseline_auc=0.6, baseline_positive_rate=0.5)
        assert check.verdict == "improved"

    def test_thin_coverage_is_flagged_in_the_note(self):
        rng = np.random.default_rng(5)
        labels = (rng.random(400) < 0.5).astype(int)
        scores = labels * 0.5 + rng.random(400) * 0.5
        check = check_performance(
            labels, scores, baseline_auc=0.7, baseline_positive_rate=0.5, coverage=0.08
        )
        assert "provisional" in check.note
        assert check.to_dict()["coverage"] == pytest.approx(0.08)


class TestPositiveRateShift:
    def test_no_shift_is_not_significant(self):
        _difference, p_value = positive_rate_shift(0.10, 0.10, 2_000)
        assert p_value == pytest.approx(1.0)

    def test_a_doubled_base_rate_is_significant(self):
        difference, p_value = positive_rate_shift(0.20, 0.10, 5_000)
        assert difference == pytest.approx(0.1)
        assert p_value < 1e-10

    def test_the_same_shift_on_a_thin_window_is_not(self):
        _difference, p_value = positive_rate_shift(0.20, 0.10, 20)
        assert p_value > 0.05  # 20 rows cannot establish a ten-point move

    def test_invalid_input_is_refused(self):
        with pytest.raises(ValueError):
            positive_rate_shift(0.2, 0.1, 0)
        with pytest.raises(ValueError):
            positive_rate_shift(0.2, 0.0, 100)
