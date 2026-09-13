"""Sequential detectors, for the metrics that arrive one window at a time.

Fixed-window tests answer "is this window different from the reference".  They are the right
tool for a batch scoring job and the wrong one for a slow ramp: each window looks acceptable on
its own while the level walks away.  Sequential detectors accumulate evidence across windows
instead, which is what catches gradual drift early - at the cost of a false-alarm rate that has
to be tuned rather than derived.

Two are implemented, both online and O(1) per update:

* :class:`PageHinkley` - cumulative-sum test with an allowance ``delta``.  Detects a change in
  mean in either direction, and resets after firing so it can find the next one.
* :class:`EWMAMonitor` - exponentially weighted mean against control limits derived from the
  reference standard deviation, which is the classic control-chart formulation and is easier to
  explain to a stakeholder than a CUSUM.

Neither replaces the windowed tests; the evaluation harness runs all of them side by side and
reports detection delay against false alarms, which is the only honest way to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageHinkley:
    """Two-sided Page-Hinkley change detector on a stream of scalars.

    ``delta`` is the drift you are willing to tolerate before the statistic starts accumulating;
    ``threshold`` is how much accumulated evidence triggers an alarm.  Setting ``delta`` to zero
    makes the test fire on any persistent bias, including a harmless one.
    """

    delta: float = 0.005
    threshold: float = 0.25
    min_samples: int = 5
    n: int = 0
    mean: float = 0.0
    _up: float = 0.0
    _up_min: float = 0.0
    _down: float = 0.0
    _down_min: float = 0.0
    alarms: list[int] = field(default_factory=list)
    statistic: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.delta < 0:
            raise ValueError("delta must be non-negative")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")

    def reset(self) -> None:
        """Forget the accumulated evidence but keep the alarm history."""
        self.n = 0
        self.mean = 0.0
        self._up = self._up_min = self._down = self._down_min = 0.0
        self.statistic = 0.0

    def update(self, value: float, index: int | None = None) -> bool:
        """Feed one observation.  Returns True when an alarm fires (and then resets)."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        self.n += 1
        self.mean += (value - self.mean) / self.n

        self._up += value - self.mean - self.delta
        self._up_min = min(self._up_min, self._up)
        self._down += self.mean - self.delta - value
        self._down_min = min(self._down_min, self._down)
        self.statistic = max(self._up - self._up_min, self._down - self._down_min)

        if self.n >= self.min_samples and self.statistic > self.threshold:
            self.alarms.append(self.n - 1 if index is None else int(index))
            self.reset()
            return True
        return False


@dataclass
class EWMAMonitor:
    """Exponentially weighted moving average against control limits.

    The steady-state standard deviation of an EWMA with weight ``lam`` is
    ``sigma * sqrt(lam / (2 - lam))``, so a three-sigma limit on the raw metric would be far too
    wide for the smoothed one.  Getting that factor wrong is the usual reason a control chart
    never fires.
    """

    target: float
    sigma: float
    lam: float = 0.2
    limit_sigmas: float = 3.0
    min_samples: int = 3
    n: int = 0
    value: float = float("nan")
    alarms: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.lam <= 1:
            raise ValueError("lam must lie in (0, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.limit_sigmas <= 0:
            raise ValueError("limit_sigmas must be positive")
        self.value = float(self.target)

    @property
    def control_limit(self) -> float:
        return float(
            self.limit_sigmas * self.sigma * np.sqrt(self.lam / (2.0 - self.lam))
        )

    def update(self, observation: float, index: int | None = None) -> bool:
        observation = float(observation)
        if not np.isfinite(observation):
            raise ValueError("observation must be finite")
        self.n += 1
        self.value = self.lam * observation + (1.0 - self.lam) * self.value
        if self.n >= self.min_samples and abs(self.value - self.target) > self.control_limit:
            self.alarms.append(self.n - 1 if index is None else int(index))
            self.value = float(self.target)  # re-centre so one shift is one alarm
            return True
        return False


def first_alarm(stream, detector) -> int | None:
    """Index of the first alarm raised while replaying ``stream`` through ``detector``."""
    for index, value in enumerate(stream):
        if detector.update(value, index):
            return index
    return None


def alarm_indices(stream, detector) -> list[int]:
    """Every alarm index from replaying a stream - used to count false alarms."""
    for index, value in enumerate(stream):
        detector.update(value, index)
    return list(detector.alarms)
