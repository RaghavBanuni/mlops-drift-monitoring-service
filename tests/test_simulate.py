"""The generator is test infrastructure, so it gets tested too.

If ``concept`` accidentally moved a feature, the headline claim of this repository - that no
feature-drift detector can see concept drift - would be quietly false, and every downstream
result would inherit the error.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.metrics import ks_test
from driftwatch.performance import roc_auc
from driftwatch.simulate import (
    FEATURES,
    NUMERIC_FEATURES,
    SCENARIOS,
    ScenarioConfig,
    build_scenario,
    evaluate_all,
    evaluate_detectors,
    score,
)

CONFIG = ScenarioConfig(n_reference=1_000, n_window=300, n_windows=6, change_at=3, seed=5)


class TestConfiguration:
    def test_incoherent_configurations_are_refused(self):
        with pytest.raises(ValueError):
            ScenarioConfig(n_reference=100).validate()
        with pytest.raises(ValueError):
            ScenarioConfig(n_window=10).validate()
        with pytest.raises(ValueError):
            ScenarioConfig(change_at=25, n_windows=20).validate()

    def test_an_unknown_scenario_names_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown scenario"):
            build_scenario("nonsense", CONFIG)

    def test_every_advertised_scenario_builds(self):
        assert len(SCENARIOS) == 7
        for name in SCENARIOS:
            built = build_scenario(name, CONFIG)
            assert len(built.windows) == CONFIG.n_windows
            assert len(built.reference) == CONFIG.n_reference
            assert built.description
            for column in (*FEATURES, "row_id", "prediction", "label"):
                assert column in built.reference.columns

    def test_only_the_scenarios_with_a_real_change_carry_a_change_point(self):
        assert build_scenario("stable", CONFIG).change_at is None
        assert build_scenario("benign_seasonal", CONFIG).change_at is None
        assert build_scenario("sudden_covariate", CONFIG).change_at == CONFIG.change_at

    def test_row_ids_are_unique_across_the_whole_run(self):
        built = build_scenario("stable", CONFIG)
        ids = list(built.reference["row_id"])
        for window in built.windows:
            ids.extend(window["row_id"])
        assert len(set(ids)) == len(ids)  # the label join depends on this


class TestTheDeployedModel:
    def test_the_model_is_frozen_and_deterministic(self):
        built = build_scenario("stable", CONFIG)
        window = built.windows[0]
        assert score(window) == pytest.approx(window["prediction"].to_numpy())
        assert score(window) == pytest.approx(score(window))

    def test_predictions_are_probabilities_that_actually_rank(self):
        built = build_scenario("stable", CONFIG)
        assert built.reference["prediction"].between(0, 1).all()
        assert roc_auc(built.reference["label"], built.reference["prediction"]) > 0.6

    def test_a_null_feature_is_imputed_rather_than_crashing_the_model(self):
        # the frozen median imputation is what makes a null spike invisible to the model and
        # visible only to monitoring
        built = build_scenario("null_spike", CONFIG)
        after = built.windows[-1]
        assert after["credit_score"].isna().mean() > 0.2
        assert np.isfinite(score(after)).all()


class TestScenarioContent:
    def test_concept_drift_leaves_every_feature_alone(self):
        """The claim the whole repository rests on, checked rather than asserted."""
        built = build_scenario("concept", CONFIG)
        before, after = built.windows[0], built.windows[-1]

        for name in NUMERIC_FEATURES:
            _statistic, p_value = ks_test(before[name], after[name])
            assert p_value > 0.01, f"{name} moved in a concept-drift scenario"
        assert set(before["region"]) == set(after["region"])

        # the relationship, however, has changed: the deployed model ranks worse afterwards
        assert roc_auc(after["label"], after["prediction"]) < roc_auc(
            before["label"], before["prediction"]
        )

    def test_a_sudden_covariate_shift_moves_only_what_it_claims_to(self):
        built = build_scenario("sudden_covariate", CONFIG)
        before, after = built.windows[0], built.windows[-1]
        assert after["income"].mean() > before["income"].mean()
        assert after["utilisation"].mean() > before["utilisation"].mean()
        _statistic, tenure = ks_test(before["tenure_months"], after["tenure_months"])
        assert tenure > 0.01

    def test_a_gradual_shift_creeps(self):
        built = build_scenario("gradual_covariate", CONFIG)
        means = [window["income"].mean() for window in built.windows]
        assert means[-1] > means[CONFIG.change_at]
        # no single step is large: this is what defeats a window-on-window comparison
        steps = np.diff(means[CONFIG.change_at :]) / np.mean(means)
        assert np.max(np.abs(steps)) < 0.2

    def test_a_new_category_appears_only_after_the_change(self):
        built = build_scenario("new_category", CONFIG)
        assert "overseas" not in set(built.reference["region"])
        assert "overseas" not in set(built.windows[0]["region"])
        share = (built.windows[-1]["region"] == "overseas").mean()
        assert 0.05 < share < 0.2

    def test_the_seasonal_scenario_oscillates_without_a_trend(self):
        built = build_scenario("benign_seasonal", ScenarioConfig(
            n_reference=1_000, n_window=300, n_windows=12, change_at=6, seed=5
        ))
        web = [float((window["channel"] == "web").mean()) for window in built.windows]
        assert max(web) - min(web) > 0.05  # the mix really does move
        assert abs(np.mean(web[:6]) - np.mean(web[6:])) < 0.1  # but it goes nowhere

    def test_the_same_seed_reproduces_the_same_data(self):
        first = build_scenario("sudden_covariate", CONFIG)
        second = build_scenario("sudden_covariate", CONFIG)
        assert first.windows[2]["income"].to_numpy() == pytest.approx(
            second.windows[2]["income"].to_numpy()
        )


class TestEvaluationHarness:
    @pytest.fixture(scope="class")
    def stable_table(self):
        return evaluate_detectors("stable", CONFIG)

    @pytest.fixture(scope="class")
    def shifted_table(self):
        return evaluate_detectors("sudden_covariate", CONFIG)

    def test_the_table_reports_both_halves_of_the_trade(self, shifted_table):
        assert set(shifted_table.columns) == {
            "scenario",
            "detector",
            "false_alarm_windows",
            "false_alarm_rate",
            "detected",
            "delay_windows",
        }
        assert len(shifted_table) >= 6  # several rules, scored the same way
        assert shifted_table["detector"].is_unique

    def test_a_scenario_with_no_change_has_no_delay_to_report(self, stable_table):
        assert stable_table["detected"].isna().all()
        assert stable_table["delay_windows"].isna().all()
        assert (stable_table["false_alarm_windows"] >= 0).all()

    def test_confirmation_can_only_reduce_false_alarms(self, stable_table):
        policy = stable_table.set_index("detector")["false_alarm_windows"]
        raw = policy["driftwatch policy (effect size + adaptive floor + FDR)"]
        confirmed = policy["driftwatch policy + 2-of-3 confirmation"]
        assert confirmed <= raw

    def test_the_disciplined_rule_is_no_noisier_than_uncorrected_significance(self, stable_table):
        policy = stable_table.set_index("detector")["false_alarm_windows"]
        assert (
            policy["driftwatch policy + 2-of-3 confirmation"]
            <= policy["KS per feature, alpha=0.05 (no correction)"]
        )

    def test_a_real_shift_is_eventually_detected(self, shifted_table):
        detected = shifted_table.set_index("detector")["detected"]
        assert bool(detected["driftwatch policy (effect size + adaptive floor + FDR)"])

    def test_evaluate_all_stacks_the_scenarios(self):
        frame = evaluate_all(CONFIG, scenarios=("stable", "concept"))
        assert set(frame["scenario"]) == {"stable", "concept"}

    def test_only_the_labelled_detector_sees_concept_drift(self):
        """The negative result that justifies measuring outcomes at all."""
        frame = evaluate_detectors("concept", CONFIG).set_index("detector")
        assert not bool(frame.loc["driftwatch policy (effect size + adaptive floor + FDR)", "detected"])
        assert bool(frame.loc["labelled AUC drop > 3 points (needs outcomes)", "detected"])
