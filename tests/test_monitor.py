"""End-to-end behaviour: several windows, delayed labels, and the reading at the end.

These are the tests that would catch a regression a unit test cannot see - a detector that only
works on the first window, an alert manager whose state never resets, or a reading that says
"retrain" when the outcomes have not arrived yet.
"""

from __future__ import annotations

import pytest

from driftwatch.alerts import AlertPolicy
from driftwatch.baseline import fit_baseline
from driftwatch.monitor import Monitor, MonitorConfig
from driftwatch.simulate import FEATURES, ScenarioConfig, build_scenario

CONFIG = ScenarioConfig(n_reference=1_200, n_window=400, n_windows=8, change_at=4, seed=11)
LAG = 1  # outcomes for window i arrive during window i + 1


def _fit(scenario):
    return fit_baseline(
        scenario.reference,
        features=list(FEATURES),
        prediction_column="prediction",
        target_column="label",
        model_version="replay-1",
    )


def _replay(scenario, last_windows: int | None = 3) -> Monitor:
    monitor = Monitor(
        _fit(scenario),
        alert_policy=AlertPolicy(confirm_windows=2, of_windows=3),
        config=MonitorConfig(prediction_column="prediction", id_column="row_id"),
    )
    for index, window in enumerate(scenario.windows):
        monitor.ingest(window, index)
        matured = index - LAG
        if matured >= 0:
            source = scenario.windows[matured]
            monitor.add_labels(source["row_id"], source["label"], index)
        if monitor.labels.matched:
            monitor.quality(window=index, last_windows=last_windows)
    return monitor


def _drift_alerts(monitor: Monitor):
    return [alert for alert in monitor.alerts.raised if alert.kind.startswith("drift")]


@pytest.fixture(scope="module")
def stable_run():
    return _replay(build_scenario("stable", CONFIG))


@pytest.fixture(scope="module")
def shift_run():
    return _replay(build_scenario("sudden_covariate", CONFIG))


@pytest.fixture(scope="module")
def concept_run():
    return _replay(build_scenario("concept", CONFIG))


class TestQuietPipeline:
    def test_eight_stable_windows_page_nobody(self, stable_run):
        assert _drift_alerts(stable_run) == []
        assert stable_run.windows_seen == CONFIG.n_windows

    def test_almost_every_window_reads_as_stable(self, stable_run):
        verdicts = list(stable_run.history()["verdict"])
        assert verdicts.count("stable") >= CONFIG.n_windows - 2
        assert "action_required" not in verdicts

    def test_the_prediction_stream_stays_in_control(self, stable_run):
        assert stable_run.prediction_alarms == []

    def test_the_reading_is_stable_and_says_so(self, stable_run):
        reading = stable_run.interpret()
        assert reading["case"] == "stable"
        assert reading["drifted_features"] == []
        assert "no action" in reading["action"]

    def test_history_and_summary_are_reportable(self, stable_run):
        history = stable_run.history()
        assert len(history) == CONFIG.n_windows
        assert set(history.columns) == {
            "window",
            "rows",
            "verdict",
            "flagged",
            "schema_blocking",
            "worst_feature",
            "worst_effect",
            "alerts",
        }

        summary = stable_run.summary()
        assert summary["model_version"] == "replay-1"
        assert summary["features"] == len(FEATURES)
        assert summary["windows_seen"] == CONFIG.n_windows
        assert summary["coverage"]["coverage"] > 0.5
        assert summary["reading"]["case"] == "stable"
        assert summary["alerting"]["raised"] == 0


class TestCovariateShift:
    def test_the_change_is_caught_and_not_before_it_happened(self, shift_run):
        alerts = _drift_alerts(shift_run)
        assert alerts, "a 0.8 sd shift in two features must eventually page"
        first = min(alert.window for alert in alerts)
        assert first >= CONFIG.change_at  # no alarm before anything happened
        assert first <= CONFIG.change_at + 2  # and confirmation costs at most a window or two

    def test_the_named_features_are_the_ones_that_moved(self, shift_run):
        scopes = {alert.scope for alert in _drift_alerts(shift_run)}
        assert {"income", "utilisation"} & scopes
        assert "tenure_months" not in scopes
        assert "region" not in scopes

    def test_the_reading_pairs_inputs_with_measured_quality(self, shift_run):
        reading = shift_run.interpret()
        assert reading["case"] in (
            "covariate_shift_absorbed",
            "population_moved_and_model_followed",
        )
        assert reading["drifted_features"]
        assert reading["quality"]["verdict"] in ("stable", "degraded", "improved")

    def test_the_worst_feature_is_recorded_per_window(self, shift_run):
        history = shift_run.history()
        after = history[history["window"] >= CONFIG.change_at]
        assert after["worst_effect"].max() > 0.1
        assert after["flagged"].max() >= 1


class TestConceptDrift:
    def test_no_input_drift_is_reported_because_none_happened(self, concept_run):
        assert _drift_alerts(concept_run) == []
        assert concept_run.reports[-1].flagged == ()

    def test_the_measured_quality_falls(self, concept_run):
        check = concept_run.checks[-1]
        assert check.verdict == "degraded"
        assert check.auc < check.baseline_auc - 0.03
        assert check.auc_high < check.baseline_auc  # beyond this window's sampling noise

    def test_the_reading_names_concept_drift_and_asks_for_a_retrain(self, concept_run):
        reading = concept_run.interpret()
        assert reading["case"] == "concept_drift"
        assert reading["drifted_features"] == []
        assert "retrain" in reading["action"]


class TestSchemaEvents:
    def test_an_unseen_level_is_escalated_without_needing_significance(self):
        monitor = _replay(build_scenario("new_category", CONFIG))
        after = monitor.reports[CONFIG.change_at + 1]
        region = next(item for item in after.features if item.feature == "region")

        assert region.severity in ("warn", "alert")
        assert "never saw" in region.note
        assert region.detail["unseen_share"] > 0.05
        assert any(alert.scope == "region" for alert in monitor.alerts.raised)

    def test_a_null_spike_is_caught_by_the_schema_not_by_psi(self):
        """The distinction the schema layer exists for.

        The surviving rows are still distributed exactly as before, so PSI on them is silent -
        and would stay silent if the column went 90% null.  The data-quality check is what sees
        it, which is why it runs first and why its verdict is reported separately.
        """
        monitor = _replay(build_scenario("null_spike", CONFIG))
        after = monitor.reports[CONFIG.change_at + 1]

        assert any(
            issue.feature == "credit_score" and issue.issue == "null_rate_spike"
            for issue in after.schema.issues
        )
        assert "credit_score" not in {item.feature for item in after.flagged}
        assert after.verdict == "investigate"


class TestLabelsAndReadiness:
    def test_the_reading_refuses_to_guess_before_outcomes_arrive(self):
        scenario = build_scenario("sudden_covariate", CONFIG)
        monitor = Monitor(
            _fit(scenario),
            config=MonitorConfig(prediction_column="prediction", id_column="row_id"),
        )
        for index, window in enumerate(scenario.windows):
            monitor.ingest(window, index)

        reading = monitor.interpret()
        assert reading["case"] in (
            "inputs_drifted_outcomes_pending",
            "inputs_stable_outcomes_pending",
        )
        assert reading["quality"] is None
        assert "not matured" in reading["action"] or "collecting" in reading["action"]

    def test_no_data_is_its_own_answer(self, stable_run):
        empty = Monitor(stable_run.baseline)
        assert empty.interpret() == {
            "case": "no_data",
            "action": "ingest at least one window first",
        }

    def test_coverage_reflects_the_label_lag(self, stable_run):
        coverage = stable_run.coverage()
        assert coverage["labelled"] == (CONFIG.n_windows - LAG) * CONFIG.n_window
        assert coverage["awaiting_label"] == LAG * CONFIG.n_window
        assert coverage["median_lag_windows"] == float(LAG)
        assert coverage["labels_arrived_too_late"] == 0

    def test_outcomes_for_unknown_rows_are_counted_not_dropped(self, stable_run):
        outcome = stable_run.add_labels(["not-a-row-id"], [1], window=99)
        assert outcome["matched"] == 0
        assert outcome["labels_arrived_too_late"] >= 1

    def test_a_baseline_without_outcomes_cannot_answer_quality_questions(self, stable_run):
        thin = fit_baseline(
            build_scenario("stable", CONFIG).reference,
            features=list(FEATURES),
            prediction_column="prediction",
        )
        monitor = Monitor(thin)
        with pytest.raises(ValueError, match="no reference performance"):
            monitor.quality()

    def test_segment_coverage_is_available_when_a_segment_is_declared(self):
        scenario = build_scenario("stable", CONFIG)
        monitor = Monitor(
            _fit(scenario),
            config=MonitorConfig(
                prediction_column="prediction", id_column="row_id", segment_column="channel"
            ),
        )
        monitor.ingest(scenario.windows[0], 0)
        source = scenario.windows[0]
        monitor.add_labels(source["row_id"], source["label"], 1)

        segments = monitor.segments()
        assert set(segments["segment"]) == {"branch", "web", "broker"}
        assert segments["share_of_labelled"].sum() == pytest.approx(1.0, abs=1e-3)


class TestGuards:
    def test_a_non_frame_is_refused(self, stable_run):
        with pytest.raises(TypeError):
            stable_run.ingest({"income": [1, 2, 3]})

    def test_an_unknown_window_is_an_error_not_an_empty_reading(self, stable_run):
        with pytest.raises(KeyError):
            stable_run.interpret(window=99)

    def test_incoherent_configuration_is_refused(self, stable_run):
        with pytest.raises(ValueError):
            Monitor(stable_run.baseline, config=MonitorConfig(min_labelled=5))
        with pytest.raises(ValueError):
            Monitor(stable_run.baseline, config=MonitorConfig(quality_drop=0.9))

    def test_a_window_can_be_ingested_without_a_prediction_column(self, stable_run):
        scenario = build_scenario("stable", CONFIG)
        monitor = Monitor(stable_run.baseline, config=MonitorConfig(prediction_column=None))
        outcome = monitor.ingest(scenario.windows[0].drop(columns=["prediction"]), 0)

        assert outcome["verdict"] in ("stable", "investigate", "action_required")
        assert monitor.coverage()["labelled"] == 0  # nothing to join outcomes to
