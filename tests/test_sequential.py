"""Sequential detectors: they must catch a ramp without firing on stationary noise."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.sequential import EWMAMonitor, PageHinkley, alarm_indices, first_alarm


class TestPageHinkley:
    def test_stationary_noise_does_not_fire(self):
        stream = np.random.default_rng(0).normal(0.0, 1.0, 300)
        detector = PageHinkley(delta=0.5, threshold=8.0, min_samples=10)
        assert first_alarm(stream, detector) is None
        assert detector.alarms == []

    def test_a_step_change_fires_shortly_after_it_happens(self):
        rng = np.random.default_rng(1)
        stream = np.concatenate([rng.normal(0, 0.5, 60), rng.normal(3, 0.5, 60)])
        index = first_alarm(stream, PageHinkley(delta=0.2, threshold=3.0, min_samples=5))
        assert index is not None
        assert 60 <= index <= 70

    def test_a_downward_step_is_caught_too(self):
        rng = np.random.default_rng(2)
        stream = np.concatenate([rng.normal(0, 0.5, 60), rng.normal(-3, 0.5, 60)])
        index = first_alarm(stream, PageHinkley(delta=0.2, threshold=3.0, min_samples=5))
        assert index is not None and index >= 60

    def test_a_gradual_ramp_is_caught_although_no_single_step_is_large(self):
        stream = np.concatenate([np.zeros(40), np.linspace(0, 4, 60)])
        index = first_alarm(stream, PageHinkley(delta=0.05, threshold=2.0, min_samples=5))
        assert index is not None and index > 40

    def test_firing_resets_so_a_second_change_is_findable(self):
        rng = np.random.default_rng(3)
        stream = np.concatenate(
            [rng.normal(0, 0.3, 40), rng.normal(3, 0.3, 40), rng.normal(6, 0.3, 40)]
        )
        detector = PageHinkley(delta=0.2, threshold=3.0, min_samples=5)
        alarms = alarm_indices(stream, detector)
        assert len(alarms) >= 2
        assert detector.statistic >= 0.0

    def test_reset_clears_the_evidence(self):
        detector = PageHinkley(delta=0.0, threshold=100.0)
        for value in range(20):
            detector.update(float(value))
        detector.reset()
        assert detector.n == 0 and detector.statistic == 0.0

    def test_invalid_configuration_is_refused(self):
        with pytest.raises(ValueError):
            PageHinkley(threshold=0.0)
        with pytest.raises(ValueError):
            PageHinkley(delta=-1.0)
        with pytest.raises(ValueError):
            PageHinkley(min_samples=0)

    def test_non_finite_input_is_refused(self):
        with pytest.raises(ValueError):
            PageHinkley().update(float("nan"))


class TestEWMA:
    def test_the_control_limit_uses_the_smoothed_variance(self):
        monitor = EWMAMonitor(target=0.0, sigma=1.0, lam=0.2, limit_sigmas=3.0)
        expected = 3.0 * np.sqrt(0.2 / 1.8)
        assert monitor.control_limit == pytest.approx(expected)
        # the naive three-sigma limit would be three times wider and would never fire
        assert monitor.control_limit < 3.0

    def test_on_target_noise_does_not_fire(self):
        stream = np.random.default_rng(4).normal(0.0, 1.0, 200)
        monitor = EWMAMonitor(target=0.0, sigma=1.0, lam=0.2, limit_sigmas=3.0)
        assert first_alarm(stream, monitor) is None

    def test_a_shift_fires(self):
        stream = np.concatenate(
            [np.random.default_rng(5).normal(0.0, 1.0, 30), np.full(30, 2.0)]
        )
        index = first_alarm(stream, EWMAMonitor(target=0.0, sigma=1.0, lam=0.3))
        assert index is not None and 30 <= index <= 40

    def test_invalid_configuration_is_refused(self):
        with pytest.raises(ValueError):
            EWMAMonitor(target=0.0, sigma=0.0)
        with pytest.raises(ValueError):
            EWMAMonitor(target=0.0, sigma=1.0, lam=0.0)
        with pytest.raises(ValueError):
            EWMAMonitor(target=0.0, sigma=1.0, limit_sigmas=0.0)
