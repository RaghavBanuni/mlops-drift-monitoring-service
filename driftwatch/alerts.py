"""Turning drift findings into a bounded number of alerts a human will still read.

A monitor that pages on every flagged feature is a monitor that gets muted, and a muted monitor
is worse than none: it costs money and provides false assurance.  Four controls are applied
here, all of them things real on-call rotations end up inventing:

* **k-of-n confirmation.**  A feature must be flagged in ``confirm_windows`` of the last
  ``of_windows`` before it pages.  One odd batch is not an incident.
* **Cooldown.**  After firing, a feature stays quiet for ``cooldown_windows`` unless it
  escalates.  Drift is persistent by nature, so without this one incident becomes fifty tickets.
* **Per-window cap.**  When a pipeline change moves thirty features at once, thirty alerts carry
  no more information than one that says "thirty features moved, worst first".
* **Incidents, not events.**  Consecutive alerts on the same feature share an incident id and
  raise its severity, which is what makes "is this getting worse" answerable.

Blocking schema failures bypass confirmation deliberately: a missing column or a column that
arrived as text is not a statistical fluctuation, and waiting three windows to mention it is
indefensible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .detect import DriftReport


@dataclass(frozen=True)
class AlertPolicy:
    confirm_windows: int = 2
    of_windows: int = 3
    cooldown_windows: int = 4
    max_alerts_per_window: int = 5
    escalate_after: int = 3

    def validate(self) -> None:
        if self.confirm_windows < 1:
            raise ValueError("confirm_windows must be at least 1")
        if self.of_windows < self.confirm_windows:
            raise ValueError("of_windows cannot be smaller than confirm_windows")
        if self.cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        if self.max_alerts_per_window < 1:
            raise ValueError("max_alerts_per_window must be at least 1")
        if self.escalate_after < 1:
            raise ValueError("escalate_after must be at least 1")


@dataclass(frozen=True)
class Alert:
    window: int
    scope: str
    severity: str
    kind: str
    effect: float | None
    reason: str
    incident: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AlertManager:
    """Stateful across windows, because every control here is about history."""

    policy: AlertPolicy = field(default_factory=AlertPolicy)
    history: dict[str, deque] = field(default_factory=dict)
    last_fired: dict[str, int] = field(default_factory=dict)
    streaks: dict[str, int] = field(default_factory=dict)
    incidents: dict[str, str] = field(default_factory=dict)
    raised: list[Alert] = field(default_factory=list)
    suppressed_by_confirmation: int = 0
    suppressed_by_cooldown: int = 0
    suppressed_by_cap: int = 0

    def __post_init__(self) -> None:
        self.policy.validate()

    def _record(self, scope: str, flagged: bool) -> int:
        window_history = self.history.setdefault(scope, deque(maxlen=self.policy.of_windows))
        window_history.append(bool(flagged))
        return sum(window_history)

    def _incident_id(self, scope: str, window: int) -> str:
        previous = self.last_fired.get(scope)
        if previous is None or window - previous > self.policy.cooldown_windows:
            self.incidents[scope] = f"{scope}:{window}"
        return self.incidents.setdefault(scope, f"{scope}:{window}")

    def process(self, report: DriftReport) -> list[Alert]:
        """Apply the policy to one drift report and return the alerts that survive it."""
        window = report.window
        candidates: list[tuple[float, Alert]] = []

        for issue in report.schema.blocking:
            # bypasses confirmation on purpose: this is a broken contract, not a fluctuation
            candidates.append(
                (
                    float("inf"),
                    Alert(
                        window=window,
                        scope=issue.feature,
                        severity="blocking",
                        kind=f"schema:{issue.issue}",
                        effect=None,
                        reason=issue.detail,
                        incident=self._incident_id(f"schema:{issue.feature}", window),
                    ),
                )
            )

        tracked = {item.feature for item in report.features}
        for item in report.features:
            hits = self._record(item.feature, item.flagged)
            if not item.flagged:
                self.streaks[item.feature] = 0
                continue
            if hits < self.policy.confirm_windows:
                self.suppressed_by_confirmation += 1
                continue

            previous = self.last_fired.get(item.feature)
            self.streaks[item.feature] = self.streaks.get(item.feature, 0) + 1
            escalated = self.streaks[item.feature] >= self.policy.escalate_after
            if (
                previous is not None
                and window - previous <= self.policy.cooldown_windows
                and not (escalated and item.severity == "alert")
            ):
                self.suppressed_by_cooldown += 1
                continue

            severity = "critical" if escalated and item.severity == "alert" else item.severity
            reason = (
                f"{item.effect_name} {item.effect:.4f} against a {item.warn_threshold:.4f} "
                f"threshold ({hits} of the last {len(self.history[item.feature])} windows"
                f", streak {self.streaks[item.feature]})"
            )
            candidates.append(
                (
                    item.effect,
                    Alert(
                        window=window,
                        scope=item.feature,
                        severity=severity,
                        kind=f"drift:{item.kind}",
                        effect=round(float(item.effect), 5),
                        reason=reason,
                        incident=self._incident_id(item.feature, window),
                    ),
                )
            )

        for scope in set(self.history) - tracked:
            self._record(scope, False)  # keep windows aligned for features not present

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        kept = [alert for _effect, alert in candidates[: self.policy.max_alerts_per_window]]
        overflow = len(candidates) - len(kept)
        if overflow > 0:
            self.suppressed_by_cap += overflow
            kept.append(
                Alert(
                    window=window,
                    scope="__digest__",
                    severity="warn",
                    kind="digest",
                    effect=None,
                    reason=(
                        f"{overflow} further features crossed their thresholds in this window; "
                        "a simultaneous move on this many features points at the pipeline, not "
                        "at the population"
                    ),
                    incident=self._incident_id("__digest__", window),
                )
            )

        for alert in kept:
            if alert.scope != "__digest__":
                self.last_fired[alert.scope] = window
        self.raised.extend(kept)
        return kept

    def stats(self) -> dict[str, int]:
        return {
            "raised": len(self.raised),
            "suppressed_by_confirmation": self.suppressed_by_confirmation,
            "suppressed_by_cooldown": self.suppressed_by_cooldown,
            "suppressed_by_cap": self.suppressed_by_cap,
        }

    def frame(self) -> pd.DataFrame:
        if not self.raised:
            return pd.DataFrame(
                columns=["window", "scope", "severity", "kind", "effect", "reason", "incident"]
            )
        return pd.DataFrame([alert.to_dict() for alert in self.raised])
