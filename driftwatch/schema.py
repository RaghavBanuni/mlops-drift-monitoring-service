"""Schema and data-quality checks, which run *before* any distribution test.

Most "model broke in production" incidents are not subtle distribution shifts.  They are a
column that arrived as a string, a join that silently filled a feature with its default, a
vendor renaming a category, or a null rate that went from 2% to 40% after an upstream deploy.
A PSI computed on a column that is now 90% nulls is not a drift measurement, it is noise with a
decimal point - so a blocking schema issue suppresses the distribution test for that feature
and says why, instead of publishing a number that looks like evidence.

Severity has an operational meaning here:

* ``blocking`` - the feature is unusable this window; the model's input is invalid.
* ``warning``  - the feature is usable but something moved that a human should see.
* ``info``     - context that changes no decision on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import Baseline, OTHER_LABEL
from .metrics import NULL_LABEL, clean_numeric, null_rate

SEVERITIES: tuple[str, ...] = ("info", "warning", "blocking")
WINDOW_SCOPE = "__window__"


@dataclass(frozen=True)
class SchemaIssue:
    feature: str
    issue: str
    severity: str
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SchemaReport:
    rows: int
    issues: tuple[SchemaIssue, ...]
    checked: tuple[str, ...]

    @property
    def blocking(self) -> tuple[SchemaIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "blocking")

    @property
    def blocked_features(self) -> set[str]:
        """Features whose distribution test must not be trusted this window."""
        return {issue.feature for issue in self.blocking if issue.feature != WINDOW_SCOPE}

    @property
    def ok(self) -> bool:
        return not self.blocking

    def frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["feature", "issue", "severity", "detail"])
        order = {name: index for index, name in enumerate(reversed(SEVERITIES))}
        frame = pd.DataFrame([issue.to_dict() for issue in self.issues])
        return frame.sort_values(
            "severity", key=lambda column: column.map(order), ignore_index=True
        )

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _numeric_view(series: pd.Series) -> tuple[np.ndarray, float]:
    """Coerce to float and report the share of non-null values that failed to parse.

    A numeric column delivered as text is the single most damaging schema failure, because
    ``pd.to_numeric(errors='coerce')`` - or its equivalent inside a feature pipeline - turns the
    whole column into nulls, the model imputes them, and nothing crashes.
    """
    non_null = series.notna().sum()
    coerced = pd.to_numeric(series, errors="coerce")
    failed = int(non_null - coerced.notna().sum())
    share = float(failed / non_null) if non_null else 0.0
    return coerced.to_numpy(dtype=float), share


def check_schema(
    frame: pd.DataFrame,
    baseline: Baseline,
    null_rate_tolerance: float = 0.05,
    null_rate_multiplier: float = 2.0,
    range_tolerance: float = 0.25,
    out_of_range_tolerance: float = 0.005,
    unseen_tolerance: float = 0.01,
    unseen_blocking: float = 0.25,
    parse_failure_tolerance: float = 0.01,
    min_rows: int = 100,
) -> SchemaReport:
    """Validate a window against the reference contract."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    issues: list[SchemaIssue] = []
    rows = len(frame)

    if rows == 0:
        return SchemaReport(
            rows=0,
            issues=(
                SchemaIssue(
                    WINDOW_SCOPE, "empty_window", "blocking", "the window contains no rows"
                ),
            ),
            checked=(),
        )
    if rows < min_rows:
        issues.append(
            SchemaIssue(
                WINDOW_SCOPE,
                "thin_window",
                "info",
                f"{rows} rows is below the {min_rows}-row minimum: drift statistics will be "
                "dominated by sampling noise, so thresholds are widened accordingly",
            )
        )

    expected = baseline.feature_names
    unexpected = [column for column in frame.columns if column not in expected]
    if unexpected:
        issues.append(
            SchemaIssue(
                WINDOW_SCOPE,
                "unexpected_columns",
                "info",
                f"columns present but not monitored: {sorted(unexpected)[:10]}",
            )
        )

    for name, reference in baseline.numeric.items():
        if name not in frame.columns:
            issues.append(
                SchemaIssue(name, "absent_column", "blocking", "the column is missing entirely")
            )
            continue
        values, parse_failure = _numeric_view(frame[name])
        if parse_failure > parse_failure_tolerance:
            issues.append(
                SchemaIssue(
                    name,
                    "type_mismatch",
                    "blocking",
                    f"{parse_failure:.1%} of non-null values are not numeric; downstream "
                    "coercion would turn them into silent nulls",
                )
            )
            continue

        current_nulls = null_rate(values)
        if current_nulls >= 1.0:
            issues.append(
                SchemaIssue(name, "all_null", "blocking", "every value in the window is null")
            )
            continue
        allowed = max(
            reference.null_rate + null_rate_tolerance,
            reference.null_rate * null_rate_multiplier,
        )
        if current_nulls > allowed:
            issues.append(
                SchemaIssue(
                    name,
                    "null_rate_spike",
                    "blocking" if current_nulls > 0.5 else "warning",
                    f"null rate {current_nulls:.1%} against {reference.null_rate:.1%} in the "
                    "reference; the drift test only sees the surviving rows",
                )
            )

        clean = clean_numeric(values)
        if clean.size == 0:
            continue
        if reference.std > 0 and np.unique(clean).size == 1:
            issues.append(
                SchemaIssue(
                    name,
                    "constant_column",
                    "blocking",
                    f"every row is {clean[0]:.6g} while the reference varied: this is the"
                    " signature of a failed join or an upstream default",
                )
            )
            continue

        pad = range_tolerance * max(reference.spread, 1e-12)
        outside = float(
            np.mean((clean < reference.minimum - pad) | (clean > reference.maximum + pad))
        )
        if outside > out_of_range_tolerance:
            issues.append(
                SchemaIssue(
                    name,
                    "out_of_range",
                    "warning",
                    f"{outside:.2%} of values fall outside the reference range "
                    f"[{reference.minimum:.4g}, {reference.maximum:.4g}]",
                )
            )
        if reference.minimum >= 0 and clean.min() < 0:
            issues.append(
                SchemaIssue(
                    name,
                    "sign_flip",
                    "warning",
                    f"negative values appeared (min {clean.min():.4g}) in a column that was "
                    "non-negative in the reference; check for sentinel codes such as -1 or -999",
                )
            )

    for name, reference in baseline.categorical.items():
        if name not in frame.columns:
            issues.append(
                SchemaIssue(name, "absent_column", "blocking", "the column is missing entirely")
            )
            continue
        keys = reference.normalise(frame[name])
        if (keys == NULL_LABEL).all():
            issues.append(
                SchemaIssue(name, "all_null", "blocking", "every value in the window is null")
            )
            continue

        current_nulls = float((keys == NULL_LABEL).mean())
        allowed = max(
            reference.null_rate + null_rate_tolerance,
            reference.null_rate * null_rate_multiplier,
        )
        if current_nulls > allowed and NULL_LABEL not in reference.categories:
            issues.append(
                SchemaIssue(
                    name,
                    "null_rate_spike",
                    "blocking" if current_nulls > 0.5 else "warning",
                    f"null rate {current_nulls:.1%} against {reference.null_rate:.1%} in the "
                    "reference",
                )
            )

        known = set(reference.categories)
        new_levels = sorted(set(keys.unique()) - known - {OTHER_LABEL})
        if new_levels:
            share = float(keys.isin(new_levels).mean())
            if share > unseen_tolerance:
                issues.append(
                    SchemaIssue(
                        name,
                        "new_categories",
                        "blocking" if share > unseen_blocking else "warning",
                        f"{share:.2%} of the window is in levels the reference never saw: "
                        f"{new_levels[:5]}",
                    )
                )

        shares = dict(zip(reference.categories, reference.shares))
        vanished = [
            category
            for category, share in shares.items()
            if share >= 0.05 and category not in set(keys.unique())
        ]
        if vanished:
            issues.append(
                SchemaIssue(
                    name,
                    "missing_level",
                    "warning",
                    f"levels worth at least 5% of the reference are absent: {vanished[:5]}",
                )
            )

    return SchemaReport(rows=rows, issues=tuple(issues), checked=tuple(expected))
