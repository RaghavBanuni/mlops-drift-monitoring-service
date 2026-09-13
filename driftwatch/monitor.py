"""The stateful object that ties a window of traffic to a decision.

:mod:`driftwatch.detect` grades one window against the baseline and forgets it.  Monitoring is
not stateless, though: confirmation needs the last few windows, labels arrive weeks after the
predictions they belong to, and the question people actually ask - "is this getting worse?" -
only exists across time.  :class:`Monitor` owns that history.

The method worth reading is :meth:`Monitor.interpret`.  Feature drift and quality decay are
reported together because neither means much alone: inputs that moved while the model held up
are not a reason to retrain, and a quality drop with stable inputs is concept drift that no
amount of feature monitoring would ever have caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alerts import Alert, AlertManager, AlertPolicy
from .baseline import Baseline
from .detect import DriftPolicy, DriftReport, scan_window
from .performance import (
    LabelBuffer,
    PerformanceCheck,
    check_performance,
    coverage_report,
    segment_coverage,
)
from .schema import check_schema
from .sequential import PageHinkley


@dataclass(frozen=True)
class MonitorConfig:
    prediction_column: str | None = "prediction"
    id_column: str | None = None
    segment_column: str | None = None
    min_labelled: int = 100
    quality_drop: float = 0.03
    prediction_shift_delta: float = 0.25
    prediction_shift_threshold: float = 3.0

    def validate(self) -> None:
        if self.min_labelled < 30:
            raise ValueError("fewer than 30 outcomes cannot support an AUC comparison")
        if not 0 < self.quality_drop < 0.5:
            raise ValueError("quality_drop is an AUC difference and must lie in (0, 0.5)")


class Monitor:
    """Windows in, graded reports and bounded alerts out, with the history kept."""

    def __init__(
        self,
        baseline: Baseline,
        policy: DriftPolicy | None = None,
        alert_policy: AlertPolicy | None = None,
        config: MonitorConfig | None = None,
    ) -> None:
        self.baseline = baseline
        self.policy = policy or DriftPolicy()
        self.policy.validate()
        self.config = config or MonitorConfig()
        self.config.validate()
        self.alerts = AlertManager(policy=alert_policy or AlertPolicy())
        self.labels = LabelBuffer()
        self.reports: list[DriftReport] = []
        self.checks: list[PerformanceCheck] = []
        self.prediction_alarms: list[int] = []
        self.windows_seen = 0

        self.prediction_monitor: PageHinkley | None = None
        if baseline.prediction is not None and baseline.prediction.std > 0:
            sigma = float(baseline.prediction.std)
            self.prediction_monitor = PageHinkley(
                delta=self.config.prediction_shift_delta * sigma,
                threshold=self.config.prediction_shift_threshold * sigma,
                min_samples=3,
            )

    # ------------------------------------------------------------------ ingest

    def _row_ids(self, frame: pd.DataFrame, window: int) -> list[str]:
        column = self.config.id_column
        if column and column in frame.columns:
            return [str(value) for value in frame[column]]
        # synthetic ids keep the label join possible even when the caller sends none, though
        # the caller then cannot supply outcomes later - which the coverage report will show
        return [f"w{window}-{index}" for index in range(len(frame))]

    def ingest(self, frame: pd.DataFrame, window: int | None = None) -> dict:
        """Score one window: schema, drift, alerts, and the prediction-shift detector."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        window = self.windows_seen if window is None else int(window)

        schema = check_schema(frame, self.baseline)
        prediction_column = self.config.prediction_column
        scored = bool(prediction_column and prediction_column in frame.columns)
        report = scan_window(
            frame,
            self.baseline,
            self.policy,
            window=window,
            schema=schema,
            prediction_column=prediction_column if scored else None,
        )
        raised = self.alerts.process(report)

        shift_alarm = False
        if scored:
            segments = (
                frame[self.config.segment_column]
                if self.config.segment_column and self.config.segment_column in frame.columns
                else None
            )
            self.labels.add_predictions(
                self._row_ids(frame, window),
                frame[prediction_column],
                window,
                segments,
            )
            mean = float(pd.to_numeric(frame[prediction_column], errors="coerce").mean())
            if self.prediction_monitor is not None and np.isfinite(mean):
                shift_alarm = self.prediction_monitor.update(mean, window)
                if shift_alarm:
                    self.prediction_alarms.append(window)

        self.reports.append(report)
        self.windows_seen = max(self.windows_seen, window + 1)
        return {
            "window": window,
            "rows": len(frame),
            "verdict": report.verdict,
            "flagged": [item.feature for item in report.flagged],
            "schema_blocking": [issue.to_dict() for issue in report.schema.blocking],
            "alerts": [alert.to_dict() for alert in raised],
            "prediction_shift_alarm": shift_alarm,
        }

    # ------------------------------------------------------------------ labels

    def add_labels(self, ids, labels, window: int | None = None) -> dict:
        """Attach outcomes that arrived later, and report how complete the picture is."""
        window = self.windows_seen if window is None else int(window)
        matched = self.labels.add_labels(ids, labels, window)
        return {"matched": matched, **coverage_report(self.labels)}

    def coverage(self) -> dict:
        return coverage_report(self.labels)

    def segments(self) -> pd.DataFrame:
        return segment_coverage(self.labels)

    # ----------------------------------------------------------------- quality

    def quality(self, window: int | None = None, last_windows: int | None = None) -> PerformanceCheck:
        """Measured performance on whatever outcomes have matured.

        ``last_windows`` restricts the evaluation to recent traffic.  Leaving it unset pools
        every labelled row, which is the stabler estimate and the slower one to react - the same
        trade as any moving average, and worth stating rather than hiding.
        """
        if self.baseline.baseline_auc is None or self.baseline.positive_rate is None:
            raise ValueError(
                "this baseline carries no reference performance; fit it with both a "
                "prediction_column and a target_column to enable quality checks"
            )
        window = self.windows_seen - 1 if window is None else int(window)
        frame = self.labels.frame()
        if last_windows is not None:
            if last_windows < 1:
                raise ValueError("last_windows must be at least 1")
            frame = frame[frame["window"] > window - last_windows]

        check = check_performance(
            frame["label"].to_numpy(dtype=float) if len(frame) else np.array([]),
            frame["score"].to_numpy(dtype=float) if len(frame) else np.array([]),
            baseline_auc=float(self.baseline.baseline_auc),
            baseline_positive_rate=float(self.baseline.positive_rate),
            window=window,
            coverage=float(coverage_report(self.labels)["coverage"]),
            min_labelled=self.config.min_labelled,
        )
        self.checks.append(check)
        return check

    # -------------------------------------------------------------- reading it

    def interpret(self, window: int | None = None) -> dict:
        """Combine input drift and measured quality into one of four readings.

        This is the whole argument of the repository in one method: a drift alert is evidence
        about the inputs, not about the model, and only the pair of signals identifies what to
        do.  When outcomes have not matured yet, that is reported as its own state rather than
        being silently treated as "quality fine".
        """
        if not self.reports:
            return {"case": "no_data", "action": "ingest at least one window first"}
        report = self.reports[-1] if window is None else self._report(window)
        drifted = bool(report.flagged) or bool(report.schema.blocking)

        check = self.checks[-1] if self.checks else None
        if check is None or check.verdict == "insufficient_labels":
            return {
                "case": "inputs_drifted_outcomes_pending" if drifted else "inputs_stable_outcomes_pending",
                "drifted_features": [item.feature for item in report.flagged],
                "quality": None if check is None else check.to_dict(),
                "action": (
                    "outcomes have not matured, so the model's quality is unknown; treat input "
                    "drift as a prompt to investigate the pipeline, not as proof of decay"
                    if drifted
                    else "nothing to act on yet; keep collecting outcomes"
                ),
            }

        degraded = check.auc < float(self.baseline.baseline_auc) - self.config.quality_drop
        if drifted and degraded:
            case, action = (
                "population_moved_and_model_followed",
                "investigate the pipeline first - a simultaneous move often means an upstream "
                "change rather than genuine population drift - then retrain",
            )
        elif drifted:
            case, action = (
                "covariate_shift_absorbed",
                "record it and leave the model alone; retraining on a moved population without "
                "a measured quality loss usually adds risk rather than removing it",
            )
        elif degraded:
            case, action = (
                "concept_drift",
                "retrain on recent labelled data: the inputs did not move, the relationship "
                "did, and no feature-drift monitor can see this",
            )
        else:
            case, action = ("stable", "no action; keep the baseline pinned to the trained model")
        return {
            "case": case,
            "drifted_features": [item.feature for item in report.flagged],
            "quality": check.to_dict(),
            "action": action,
        }

    def _report(self, window: int) -> DriftReport:
        for report in self.reports:
            if report.window == window:
                return report
        raise KeyError(f"no report for window {window}")

    def history(self) -> pd.DataFrame:
        """One row per window: what was flagged, how hard, and what got paged."""
        if not self.reports:
            return pd.DataFrame(
                columns=["window", "rows", "verdict", "flagged", "worst_feature", "worst_effect"]
            )
        rows = []
        for report in self.reports:
            graded = [item for item in report.features if not np.isnan(item.effect)]
            worst = max(graded, key=lambda item: item.effect, default=None)
            rows.append(
                {
                    "window": report.window,
                    "rows": report.rows,
                    "verdict": report.verdict,
                    "flagged": len(report.flagged),
                    "schema_blocking": len(report.schema.blocking),
                    "worst_feature": None if worst is None else worst.feature,
                    "worst_effect": None if worst is None else round(float(worst.effect), 4),
                    "alerts": sum(1 for alert in self.alerts.raised if alert.window == report.window),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        return {
            "model_version": self.baseline.model_version,
            "baseline_created_at": self.baseline.created_at,
            "baseline_rows": self.baseline.n_rows,
            "baseline_auc": self.baseline.baseline_auc,
            "features": len(self.baseline.feature_names),
            "windows_seen": self.windows_seen,
            "last_verdict": self.reports[-1].verdict if self.reports else None,
            "prediction_shift_alarms": list(self.prediction_alarms),
            "alerting": self.alerts.stats(),
            "coverage": coverage_report(self.labels),
            "reading": self.interpret(),
        }
