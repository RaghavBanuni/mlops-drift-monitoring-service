"""The CLI is the front door of the repository, so it is tested as a user would run it."""

from __future__ import annotations

import json

import pytest

from driftwatch.cli import build_parser, main
from driftwatch.simulate import FEATURES, ScenarioConfig, build_scenario

SMALL = [
    "--reference-rows", "600",
    "--window-rows", "150",
    "--windows", "4",
    "--change-at", "2",
    "--seed", "3",
]


@pytest.fixture(scope="module")
def scenario():
    return build_scenario(
        "sudden_covariate",
        ScenarioConfig(n_reference=1_000, n_window=300, n_windows=4, change_at=2, seed=9),
    )


@pytest.fixture(scope="module")
def reference_csv(scenario, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "reference.csv"
    scenario.reference.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def window_csv(scenario, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "window.csv"
    scenario.windows[-1].to_csv(path, index=False)
    return path


class TestParser:
    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_an_unknown_scenario_is_refused_before_any_work(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["evaluate", "--scenario", "bogus"])

    def test_every_subcommand_is_wired_to_a_handler(self):
        parser = build_parser()
        for command in ("decisions", "evaluate", "replay"):
            assert hasattr(parser.parse_args([command]), "handler")


class TestDecisions:
    def test_the_table_is_printed(self, capsys):
        assert main(["decisions"]) == 0
        printed = capsys.readouterr().out
        assert "concept drift" in printed
        assert "retrain" in printed


class TestBaselineCommand:
    def test_a_baseline_is_written_and_described(self, reference_csv, tmp_path, capsys):
        out = tmp_path / "baseline.json"
        code = main(
            [
                "baseline",
                "--data", str(reference_csv),
                "--out", str(out),
                "--features", *FEATURES,
                "--prediction-column", "prediction",
                "--target-column", "label",
                "--window-rows", "300",
            ]
        )
        assert code == 0
        assert out.exists()

        payload = json.loads(out.read_text())
        assert set(payload["numeric"]) >= {"income", "credit_score"}
        assert payload["baseline_auc"] is not None

        printed = capsys.readouterr().out
        assert "baseline written to" in printed
        # the noise floor is reported up front, because it is what the threshold depends on
        assert "PSI sits around" in printed

    def test_a_missing_input_file_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["baseline", "--data", str(tmp_path / "absent.csv")])


class TestScanCommand:
    @pytest.fixture
    def baseline_file(self, reference_csv, tmp_path):
        out = tmp_path / "baseline.json"
        main(
            [
                "baseline",
                "--data", str(reference_csv),
                "--out", str(out),
                "--features", *FEATURES,
                "--prediction-column", "prediction",
                "--target-column", "label",
            ]
        )
        return out

    def test_one_window_is_scored_and_explained(self, baseline_file, window_csv, capsys):
        code = main(
            [
                "scan",
                "--baseline", str(baseline_file),
                "--data", str(window_csv),
                "--window", "3",
                "--prediction-column", "prediction",
                "--id-column", "row_id",
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "schema" in printed and "drift" in printed
        assert "verdict:" in printed
        assert "income" in printed

    def test_a_first_window_cannot_confirm_anything(self, baseline_file, window_csv, capsys):
        main(["scan", "--baseline", str(baseline_file), "--data", str(window_csv)])
        assert "nothing paged" in capsys.readouterr().out

    def test_the_full_report_can_be_written_out(self, baseline_file, window_csv, tmp_path):
        destination = tmp_path / "report.json"
        main(
            [
                "scan",
                "--baseline", str(baseline_file),
                "--data", str(window_csv),
                "--json", str(destination),
            ]
        )
        report = json.loads(destination.read_text())
        assert report["verdict"] in ("stable", "investigate", "action_required")
        assert len(report["features"]) == len(FEATURES)
        assert "schema" in report

    def test_thresholds_are_tunable_from_the_command_line(self, baseline_file, window_csv, capsys):
        main(
            [
                "scan",
                "--baseline", str(baseline_file),
                "--data", str(window_csv),
                "--psi-warn", "0.9",
                "--psi-alert", "0.95",
            ]
        )
        # with the bar set absurdly high, a genuinely drifted window comes back quiet: the
        # thresholds are a policy the operator owns, not a constant baked into the detector
        printed = capsys.readouterr().out
        assert "nothing paged" in printed
        assert "action_required" not in printed


class TestEvaluateCommand:
    def test_one_scenario_is_scored(self, capsys):
        assert main(["evaluate", "--scenario", "stable", *SMALL]) == 0
        printed = capsys.readouterr().out
        assert "driftwatch policy" in printed
        assert "false_alarm_windows" in printed
        assert "it is louder" in printed  # the point of the table, printed with it

    def test_a_scenario_with_ground_truth_reports_delay(self, capsys):
        assert main(["evaluate", "--scenario", "sudden_covariate", *SMALL]) == 0
        assert "delay_windows" in capsys.readouterr().out


class TestReplayCommand:
    def test_a_full_replay_reports_a_reading(self, capsys):
        code = main(["replay", "--scenario", "concept", *SMALL, "--label-lag", "1"])
        assert code == 0
        printed = capsys.readouterr().out
        assert "reading:" in printed and "action:" in printed
        assert "coverage:" in printed
        assert "ground truth" in printed  # the injected change point, stated

    def test_a_scenario_without_a_change_point_says_nothing_about_one(self, capsys):
        assert main(["replay", "--scenario", "stable", *SMALL, "--label-lag", "1"]) == 0
        assert "ground truth" not in capsys.readouterr().out
