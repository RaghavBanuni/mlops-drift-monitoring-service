"""Alert policy: the difference between a monitor people read and one they mute."""

from __future__ import annotations

import pytest

from driftwatch.alerts import AlertManager, AlertPolicy
from driftwatch.detect import DriftPolicy, DriftReport, FeatureDrift
from driftwatch.schema import SchemaIssue, SchemaReport


def drift(feature: str, severity: str = "warn", effect: float = 0.3) -> FeatureDrift:
    return FeatureDrift(
        feature=feature,
        kind="numeric",
        rows=500,
        effect_name="psi",
        effect=effect,
        warn_threshold=0.1,
        alert_threshold=0.25,
        noise_floor=0.03,
        statistic=42.0,
        p_value=0.0001,
        q_value=0.0005,
        severity=severity,
        note="",
    )


def report(window: int, features=(), issues=()) -> DriftReport:
    return DriftReport(
        window=window,
        rows=500,
        features=tuple(features),
        schema=SchemaReport(rows=500, issues=tuple(issues), checked=()),
        policy=DriftPolicy(),
    )


class TestConfirmation:
    def test_a_single_odd_window_does_not_page(self):
        manager = AlertManager(policy=AlertPolicy(confirm_windows=2, of_windows=3))
        assert manager.process(report(0, [drift("income")])) == []
        assert manager.stats()["suppressed_by_confirmation"] == 1

    def test_a_repeat_within_the_window_pages(self):
        manager = AlertManager(policy=AlertPolicy(confirm_windows=2, of_windows=3))
        manager.process(report(0, [drift("income")]))
        alerts = manager.process(report(1, [drift("income")]))
        assert len(alerts) == 1
        assert alerts[0].scope == "income"
        assert alerts[0].kind == "drift:numeric"

    def test_isolated_flags_spread_too_thin_never_confirm(self):
        manager = AlertManager(policy=AlertPolicy(confirm_windows=2, of_windows=3))
        for window in range(9):
            features = [drift("income")] if window % 3 == 0 else []
            assert manager.process(report(window, features)) == []
        assert manager.stats()["raised"] == 0

    def test_a_quiet_window_resets_the_streak(self):
        manager = AlertManager(policy=AlertPolicy(confirm_windows=2, of_windows=2))
        manager.process(report(0, [drift("income")]))
        manager.process(report(1, []))
        assert manager.process(report(2, [drift("income")])) == []


class TestCooldownAndEscalation:
    def test_a_persistent_problem_does_not_page_every_window(self):
        manager = AlertManager(
            policy=AlertPolicy(confirm_windows=2, of_windows=3, cooldown_windows=4)
        )
        raised = [len(manager.process(report(window, [drift("income")]))) for window in range(6)]
        assert sum(raised) == 1  # one incident, not six tickets
        assert manager.stats()["suppressed_by_cooldown"] >= 3

    def test_a_worsening_problem_escalates_through_the_cooldown(self):
        manager = AlertManager(
            policy=AlertPolicy(
                confirm_windows=2, of_windows=3, cooldown_windows=10, escalate_after=3
            )
        )
        severities = []
        for window in range(5):
            for alert in manager.process(report(window, [drift("income", "alert", 0.4)])):
                severities.append(alert.severity)
        assert "alert" in severities  # the first confirmed window
        assert "critical" in severities  # and the escalation once it keeps getting worse

    def test_an_escalating_episode_keeps_one_incident_id(self):
        manager = AlertManager(
            policy=AlertPolicy(
                confirm_windows=2, of_windows=3, cooldown_windows=10, escalate_after=3
            )
        )
        incidents = set()
        for window in range(6):
            for alert in manager.process(report(window, [drift("income", "alert", 0.4)])):
                incidents.add(alert.incident)
        assert len(incidents) == 1  # one episode, however many times it is reported

    def test_a_new_episode_after_silence_gets_a_new_incident(self):
        manager = AlertManager(
            policy=AlertPolicy(confirm_windows=1, of_windows=1, cooldown_windows=1)
        )
        first = manager.process(report(0, [drift("income")]))[0]
        for window in range(1, 6):
            manager.process(report(window, []))
        second = manager.process(report(6, [drift("income")]))[0]
        assert first.incident != second.incident


class TestVolume:
    def test_a_pipeline_wide_move_is_summarised_not_enumerated(self):
        manager = AlertManager(
            policy=AlertPolicy(confirm_windows=1, of_windows=1, max_alerts_per_window=3)
        )
        features = [drift(f"f{index}", effect=0.2 + index / 100) for index in range(8)]
        alerts = manager.process(report(0, features))

        assert len(alerts) == 4  # three worst plus one digest
        assert alerts[-1].kind == "digest"
        assert "5 further features" in alerts[-1].reason
        assert manager.stats()["suppressed_by_cap"] == 5

    def test_the_worst_offenders_are_the_ones_kept(self):
        manager = AlertManager(
            policy=AlertPolicy(confirm_windows=1, of_windows=1, max_alerts_per_window=2)
        )
        features = [
            drift("small", effect=0.15),
            drift("large", effect=0.9),
            drift("mid", effect=0.4),
        ]
        scopes = {alert.scope for alert in manager.process(report(0, features))}
        assert "large" in scopes and "mid" in scopes
        assert "small" not in scopes


class TestSchemaBypass:
    def test_a_broken_contract_pages_immediately(self):
        manager = AlertManager(policy=AlertPolicy(confirm_windows=3, of_windows=3))
        issue = SchemaIssue("income", "absent_column", "blocking", "the column is missing entirely")
        alerts = manager.process(report(0, [], [issue]))

        assert len(alerts) == 1
        assert alerts[0].severity == "blocking"
        assert alerts[0].kind == "schema:absent_column"
        assert alerts[0].effect is None

    def test_schema_alerts_sort_above_drift_alerts(self):
        manager = AlertManager(
            policy=AlertPolicy(confirm_windows=1, of_windows=1, max_alerts_per_window=5)
        )
        issue = SchemaIssue("region", "all_null", "blocking", "every value in the window is null")
        alerts = manager.process(report(0, [drift("income", effect=0.99)], [issue]))
        assert alerts[0].kind.startswith("schema")


class TestPolicyAndOutput:
    def test_incoherent_policies_are_refused(self):
        with pytest.raises(ValueError):
            AlertPolicy(confirm_windows=3, of_windows=2).validate()
        with pytest.raises(ValueError):
            AlertPolicy(confirm_windows=0).validate()
        with pytest.raises(ValueError):
            AlertPolicy(max_alerts_per_window=0).validate()

    def test_the_frame_is_usable_when_empty_and_when_full(self):
        manager = AlertManager(policy=AlertPolicy(confirm_windows=1, of_windows=1))
        empty = manager.frame()
        assert empty.empty and "incident" in empty.columns

        manager.process(report(0, [drift("income")]))
        frame = manager.frame()
        assert len(frame) == 1
        assert set(frame.columns) == {
            "window",
            "scope",
            "severity",
            "kind",
            "effect",
            "reason",
            "incident",
        }
