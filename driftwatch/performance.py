"""Performance monitoring when labels arrive late - which is always.

Data drift is what monitoring dashboards show because it needs no labels.  Model *quality* is
what anyone actually cares about, and it needs outcomes that arrive days or weeks after the
prediction: a fraud chargeback at 45 days, a churn event at the end of the month, a repayment
at the next due date.

Three consequences are handled here explicitly:

* **Coverage is part of the metric.**  An AUC computed on the 8% of rows already labelled is
  reported next to that 8%, because it is a biased sample: fast-resolving cases are not a
  random subset.  A number without its coverage invites a retrain on nothing.
* **A drop must beat sampling noise.**  Window AUCs bounce by several points at realistic
  volumes.  Comparison against the baseline uses a bootstrap interval, not a bare threshold.
* **Prediction drift is the leading indicator, not the verdict.**  It is available immediately
  and it moves for benign reasons (a marketing campaign changes the input mix); labels are the
  only thing that closes the loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, normal_sf


def roc_auc(y_true, y_score) -> float:
    """AUC via the Mann-Whitney U statistic, with midranks for ties.

    The rank formulation is exact and O(n log n); the trapezoid-over-thresholds version people
    write by hand quietly mishandles ties, which matters for models that emit coarse scores.
    """
    labels = np.asarray(y_true, dtype=float).ravel()
    scores = np.asarray(y_score, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length")
    if labels.size == 0:
        return float("nan")
    if not np.isin(np.unique(labels), (0.0, 1.0)).all():
        raise ValueError("labels must be binary 0/1")
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # AUC is undefined on a single-class window - say so, do not fake it
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier_score(y_true, y_prob) -> float:
    """Mean squared error of the probabilities: sensitive to calibration, not just ranking."""
    labels = np.asarray(y_true, dtype=float).ravel()
    probabilities = np.asarray(y_prob, dtype=float).ravel()
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same length")
    if labels.size == 0:
        return float("nan")
    return float(np.mean((probabilities - labels) ** 2))


def accuracy_at(y_true, y_prob, threshold: float = 0.5) -> float:
    labels = np.asarray(y_true, dtype=float).ravel()
    probabilities = np.asarray(y_prob, dtype=float).ravel()
    if labels.size == 0:
        return float("nan")
    return float(((probabilities >= threshold).astype(float) == labels).mean())


@dataclass
class LabelBuffer:
    """Predictions waiting for their outcome, joined by id when the label lands.

    ``maxlen`` bounds memory: a service that keeps every prediction forever is a slow leak, and
    a prediction whose label never arrives is itself worth counting - :attr:`unmatched` makes
    that visible instead of letting it vanish.
    """

    maxlen: int = 100_000
    _pending: dict[str, dict] = field(default_factory=dict)
    _order: deque = field(default_factory=deque)
    matched: list[dict] = field(default_factory=list)
    late_labels: int = 0

    def add_predictions(
        self, ids, scores, window: int, segments=None
    ) -> None:
        ids = [str(value) for value in ids]
        scores = np.asarray(scores, dtype=float).ravel()
        if len(ids) != scores.size:
            raise ValueError("ids and scores must have the same length")
        if segments is not None and len(segments) != len(ids):
            raise ValueError("segments must align with ids")
        for position, key in enumerate(ids):
            self._pending[key] = {
                "id": key,
                "score": float(scores[position]),
                "window": int(window),
                "segment": None if segments is None else str(segments[position]),
            }
            self._order.append(key)
        while len(self._order) > self.maxlen:
            self._pending.pop(self._order.popleft(), None)

    def add_labels(self, ids, labels, window: int) -> int:
        """Attach outcomes.  Returns how many were matched; unknown ids are counted as late."""
        ids = [str(value) for value in ids]
        labels = np.asarray(labels, dtype=float).ravel()
        if len(ids) != labels.size:
            raise ValueError("ids and labels must have the same length")
        joined = 0
        for position, key in enumerate(ids):
            record = self._pending.pop(key, None)
            if record is None:
                # the prediction was already evicted, or the id is unknown: either way this
                # label cannot be used, and pretending otherwise would bias the metric
                self.late_labels += 1
                continue
            record["label"] = float(labels[position])
            record["label_window"] = int(window)
            record["lag"] = int(window) - record["window"]
            self.matched.append(record)
            joined += 1
        return joined

    @property
    def unmatched(self) -> int:
        return len(self._pending)

    def frame(self) -> pd.DataFrame:
        columns = ["id", "score", "window", "segment", "label", "label_window", "lag"]
        if not self.matched:
            return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
        return pd.DataFrame(self.matched)[columns]


def coverage_report(buffer: LabelBuffer) -> dict[str, float]:
    """How much of the traffic has resolved, and how long it took."""
    frame = buffer.frame()
    resolved = len(frame)
    total = resolved + buffer.unmatched
    return {
        "labelled": resolved,
        "awaiting_label": buffer.unmatched,
        "coverage": round(resolved / total, 4) if total else 0.0,
        "median_lag_windows": float(frame["lag"].median()) if resolved else float("nan"),
        "labels_arrived_too_late": buffer.late_labels,
    }


def segment_coverage(buffer: LabelBuffer) -> pd.DataFrame:
    """Label coverage per segment - the check that catches differential label delay.

    If one segment resolves in days and another in months, then "current AUC" is really "AUC on
    the fast segment", and a retraining decision made on it inherits that bias.
    """
    frame = buffer.frame()
    if frame.empty:
        return pd.DataFrame(columns=["segment", "labelled", "mean_lag", "share_of_labelled"])
    grouped = (
        frame.groupby(frame["segment"].fillna("unknown"), dropna=False)
        .agg(labelled=("label", "size"), mean_lag=("lag", "mean"))
        .reset_index()
        .rename(columns={"segment": "segment"})
    )
    grouped["share_of_labelled"] = (grouped["labelled"] / grouped["labelled"].sum()).round(4)
    grouped["mean_lag"] = grouped["mean_lag"].round(3)
    return grouped


@dataclass(frozen=True)
class PerformanceCheck:
    """One window's answer to "has quality moved, by more than noise?"."""

    window: int
    labelled: int
    coverage: float
    auc: float
    baseline_auc: float
    auc_low: float
    auc_high: float
    brier: float
    positive_rate: float
    baseline_positive_rate: float
    verdict: str
    note: str

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "labelled": self.labelled,
            "coverage": round(self.coverage, 4),
            "auc": None if np.isnan(self.auc) else round(self.auc, 4),
            "baseline_auc": None if np.isnan(self.baseline_auc) else round(self.baseline_auc, 4),
            "auc_ci": [
                None if np.isnan(self.auc_low) else round(self.auc_low, 4),
                None if np.isnan(self.auc_high) else round(self.auc_high, 4),
            ],
            "brier": None if np.isnan(self.brier) else round(self.brier, 4),
            "positive_rate": round(self.positive_rate, 4),
            "baseline_positive_rate": round(self.baseline_positive_rate, 4),
            "verdict": self.verdict,
            "note": self.note,
        }


def check_performance(
    labels,
    scores,
    baseline_auc: float,
    baseline_positive_rate: float,
    window: int = 0,
    coverage: float = 1.0,
    min_labelled: int = 100,
    seed: int = 0,
    n_boot: int = 300,
) -> PerformanceCheck:
    """Compare a window's quality against the baseline, with an interval rather than a threshold.

    The verdict is one of ``insufficient_labels``, ``stable``, ``degraded`` or ``improved``.  A
    degradation is only declared when the *upper* end of the bootstrap interval sits below the
    baseline, so a two-point wobble on 300 labelled rows does not page anyone.
    """
    labels = np.asarray(labels, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length")

    positive_rate = float(labels.mean()) if labels.size else float("nan")
    if labels.size < min_labelled or len(np.unique(labels)) < 2:
        return PerformanceCheck(
            window=window,
            labelled=int(labels.size),
            coverage=coverage,
            auc=float("nan"),
            baseline_auc=baseline_auc,
            auc_low=float("nan"),
            auc_high=float("nan"),
            brier=float("nan"),
            positive_rate=positive_rate if labels.size else 0.0,
            baseline_positive_rate=baseline_positive_rate,
            verdict="insufficient_labels",
            note=(
                f"{labels.size} labelled rows and "
                f"{len(np.unique(labels))} class(es): no defensible quality estimate yet"
            ),
        )

    paired = np.column_stack([labels, scores])
    auc = roc_auc(labels, scores)
    low, high = bootstrap_ci(
        paired,
        lambda sample: roc_auc(sample[:, 0], sample[:, 1]),
        n_boot=n_boot,
        seed=seed,
    )

    if not np.isnan(high) and high < baseline_auc:
        verdict = "degraded"
        note = (
            f"AUC {auc:.3f} (CI {low:.3f}-{high:.3f}) is below the baseline {baseline_auc:.3f} "
            "by more than this window's sampling noise"
        )
    elif not np.isnan(low) and low > baseline_auc:
        verdict = "improved"
        note = f"AUC {auc:.3f} is above the baseline {baseline_auc:.3f}"
    else:
        verdict = "stable"
        note = (
            f"AUC {auc:.3f} with CI {low:.3f}-{high:.3f} spans the baseline "
            f"{baseline_auc:.3f}: no measurable change"
        )
    if coverage < 0.5:
        note += f"; only {coverage:.0%} of the window is labelled, so treat this as provisional"

    return PerformanceCheck(
        window=window,
        labelled=int(labels.size),
        coverage=coverage,
        auc=auc,
        baseline_auc=baseline_auc,
        auc_low=low,
        auc_high=high,
        brier=brier_score(labels, scores),
        positive_rate=positive_rate,
        baseline_positive_rate=baseline_positive_rate,
        verdict=verdict,
        note=note,
    )


def positive_rate_shift(
    current_rate: float, baseline_rate: float, n: int
) -> tuple[float, float]:
    """``(difference, p_value)`` for a change in the observed positive rate.

    A moving base rate is the cheapest concept-drift signal available: it needs one column and
    it is what makes a fixed decision threshold silently wrong. The test is a two-sided normal
    approximation on the baseline proportion.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must lie in (0, 1)")
    standard_error = float(np.sqrt(baseline_rate * (1.0 - baseline_rate) / n))
    difference = float(current_rate - baseline_rate)
    z = abs(difference) / standard_error if standard_error > 0 else 0.0
    return round(difference, 5), float(min(2.0 * normal_sf(z), 1.0))
