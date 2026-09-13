"""Schema validation runs before any statistic, so these are the tests that protect the rest.

Note the deliberate asymmetry between an *empty* window and a merely *thin* one: an empty window
is blocking because there is nothing to test, while a thin window is only informational - the
detector widens its thresholds for small samples instead of refusing to look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.schema import WINDOW_SCOPE, check_schema


@pytest.fixture
def window(stable):
    return stable.windows[0].copy()


class TestCleanWindow:
    def test_a_matching_window_passes(self, window, baseline):
        report = check_schema(window, baseline)
        assert report.ok
        assert not report.blocking
        assert set(report.checked) == set(baseline.feature_names)

    def test_columns_that_are_not_monitored_are_only_noted(self, window, baseline):
        # row_id, prediction and label are present but not features: info, never a decision
        issues = [issue for issue in check_schema(window, baseline).issues]
        assert all(issue.severity == "info" for issue in issues)
        assert any(issue.issue == "unexpected_columns" for issue in issues)

    def test_report_serialises(self, window, baseline):
        payload = check_schema(window, baseline).to_dict()
        assert payload["ok"] is True
        assert payload["rows"] == len(window)
        assert isinstance(payload["issues"], list)


class TestBlockingProblems:
    def test_a_missing_column_blocks_that_feature(self, window, baseline):
        report = check_schema(window.drop(columns=["income"]), baseline)
        assert not report.ok
        assert "income" in report.blocked_features
        assert any(issue.issue == "absent_column" for issue in report.blocking)

    def test_a_numeric_column_arriving_as_text_blocks(self, window, baseline):
        window["income"] = "n/a"
        report = check_schema(window, baseline)
        assert "income" in report.blocked_features
        assert any(issue.issue == "type_mismatch" for issue in report.blocking)

    def test_an_all_null_column_blocks(self, window, baseline):
        window["credit_score"] = np.nan
        report = check_schema(window, baseline)
        assert "credit_score" in report.blocked_features

    def test_a_constant_column_blocks_as_a_failed_join(self, window, baseline):
        window["income"] = 42_000.0
        report = check_schema(window, baseline)
        assert any(issue.issue == "constant_column" for issue in report.blocking)

    def test_an_empty_window_blocks_the_whole_window(self, baseline, window):
        report = check_schema(window.iloc[0:0], baseline)
        assert any(
            issue.feature == WINDOW_SCOPE and issue.issue == "empty_window"
            for issue in report.blocking
        )
        assert report.blocked_features == set()  # the window is blocked, not any one feature

    def test_a_thin_window_is_noted_but_not_blocked(self, window, baseline):
        report = check_schema(window.head(20), baseline, min_rows=100)
        thin = [issue for issue in report.issues if issue.issue == "thin_window"]
        assert thin and thin[0].severity == "info"

    def test_a_flood_of_unseen_levels_blocks(self, window, baseline):
        window["region"] = "a_region_never_trained_on"
        report = check_schema(window, baseline)
        assert "region" in report.blocked_features


class TestWarnings:
    def test_a_null_spike_is_a_warning_before_it_is_blocking(self, window, baseline):
        values = window["credit_score"].to_numpy(dtype=float).copy()
        values[: int(0.2 * len(values))] = np.nan
        window["credit_score"] = values
        issues = [
            issue
            for issue in check_schema(window, baseline).issues
            if issue.feature == "credit_score"
        ]
        assert issues
        assert any(issue.issue == "null_rate_spike" and issue.severity == "warning" for issue in issues)

    def test_a_majority_null_column_escalates_to_blocking(self, window, baseline):
        values = window["credit_score"].to_numpy(dtype=float).copy()
        values[: int(0.7 * len(values))] = np.nan
        window["credit_score"] = values
        report = check_schema(window, baseline)
        assert "credit_score" in report.blocked_features

    def test_a_trickle_of_unseen_levels_only_warns(self, window, baseline):
        # assign through an object series: a numpy string array would truncate the new level
        regions = pd.Series(window["region"].to_numpy(), dtype="object")
        regions.iloc[: int(0.03 * len(regions))] = "overseas"
        window["region"] = regions.to_numpy()
        report = check_schema(window, baseline)

        assert "region" not in report.blocked_features
        assert any(
            issue.feature == "region" and issue.issue == "new_categories"
            and issue.severity == "warning"
            for issue in report.issues
        )

    def test_out_of_range_values_are_reported(self, window, baseline):
        values = window["utilisation"].to_numpy(dtype=float).copy()
        values[:50] = 50.0  # the reference tops out near 1.5
        window["utilisation"] = values
        report = check_schema(window, baseline)
        assert any(
            issue.feature == "utilisation" and issue.issue == "out_of_range"
            for issue in report.issues
        )

    def test_sentinel_negatives_are_reported(self, window, baseline):
        values = window["tenure_months"].to_numpy(dtype=float).copy()
        values[:20] = -999.0
        window["tenure_months"] = values
        report = check_schema(window, baseline)
        assert any(issue.issue == "sign_flip" for issue in report.issues)

    def test_a_vanished_level_is_reported(self, window, baseline):
        regions = pd.Series(window["region"].to_numpy(), dtype="object")
        window["region"] = regions.replace({"north": "south"}).to_numpy()
        report = check_schema(window, baseline)
        assert any(issue.issue == "missing_level" for issue in report.issues)

    def test_severity_ordering_puts_blocking_first(self, window, baseline):
        report = check_schema(window.drop(columns=["income"]), baseline)
        assert report.frame().iloc[0]["severity"] == "blocking"
