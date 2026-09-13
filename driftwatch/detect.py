"""Per-feature drift detection: effect size decides, p-values assist, FDR keeps the peace.

The decision rule is the opinionated part of this repo:

1. **Schema first.**  A feature with a blocking schema issue is not tested at all; a PSI on a
   column that is 90% null this week is a number, not evidence.
2. **Effect size decides.**  PSI for numeric features, total variation for categorical ones.
   Both answer "how much moved", which is the question a retraining decision needs.
3. **The threshold adapts to the window.**  PSI has a noise floor of about
   ``(bins - 1) * (1/n_ref + 1/n_cur)`` with no drift at all, so a 300-row window on ten bins
   sits near 0.06 before anything happens.  A fixed 0.1 therefore cries wolf on small windows
   and goes blind on large ones; the effective threshold is
   ``max(configured, noise_multiple * noise_floor)``.
4. **Significance only demotes.**  A large effect with no statistical support is downgraded, not
   dropped, and the note says which of the two is missing.
5. **Multiplicity is corrected once, across the window.**  Forty independent tests a day at 5%
   is a false alarm every other day, which is how monitoring gets muted; Benjamini-Hochberg
   bounds the false-discovery share among the features actually flagged.

Significance uses a chi-square test on the reference bins rather than a Kolmogorov-Smirnov test,
because the baseline stores no raw reference rows - only bin edges and masses.  That is a
deliberate trade (small, shareable, subject-data-free artefacts) and it makes the numeric test a
discretised one.  :func:`driftwatch.metrics.ks_test` is still available wherever both raw
samples exist, and the evaluation harness uses it to show what the discretisation costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import Baseline, CategoricalBaseline, NumericBaseline
from .metrics import (
    bin_counts,
    bin_shares,
    chi_square_homogeneity,
    clean_numeric,
    cramers_v,
    jensen_shannon,
    total_variation,
    unseen_share,
)
from .schema import WINDOW_SCOPE, SchemaReport, check_schema
from .stats import benjamini_hochberg

GRADED: tuple[str, ...] = ("none", "warn", "alert")
UNGRADED: tuple[str, ...] = ("skipped", "insufficient_data")
VERDICTS: tuple[str, ...] = ("stable", "investigate", "action_required")
SEVERITY_RANK: dict[str, int] = {
    "alert": 0,
    "warn": 1,
    "insufficient_data": 2,
    "skipped": 3,
    "none": 4,
}


def psi_noise_floor(n_reference: int, n_current: int, n_bins: int) -> float:
    """Expected PSI between two samples drawn from the *same* distribution.

    From the second-order expansion of the Jeffreys divergence: with ``k`` bins,
    ``E[PSI] ~ (k - 1) * (1/n_ref + 1/n_cur)``.  This is the most useful number in the module,
    and it is why "PSI above 0.1 means drift" is wrong on small windows: ten bins and 200 rows a
    side put the floor at 0.09 with nothing happening at all.
    """
    if min(n_reference, n_current) <= 0 or n_bins < 2:
        raise ValueError("sample sizes must be positive and n_bins at least 2")
    return float((n_bins - 1) * (1.0 / n_reference + 1.0 / n_current))


def tvd_noise_floor(reference_shares, n_reference: int, n_current: int) -> float:
    """Expected total variation distance under no drift.

    Each category share differs by a roughly normal error with variance
    ``p(1-p)(1/n_ref + 1/n_cur)``; the expected absolute value of such an error is
    ``sigma * sqrt(2/pi)``, and TVD is half their sum.  A 40-level column on a 500-row window
    has a floor around 0.1 - the same size as the effect people alert on.
    """
    if min(n_reference, n_current) <= 0:
        raise ValueError("sample sizes must be positive")
    p = np.asarray(reference_shares, dtype=float)
    if p.sum() <= 0:
        raise ValueError("reference shares must have positive mass")
    p = p / p.sum()
    variance = p * (1.0 - p) * (1.0 / n_reference + 1.0 / n_current)
    return float(0.5 * np.sum(np.sqrt(2.0 * variance / np.pi)))


def psi_from_shares(reference_shares, current_shares) -> float:
    """PSI between two already-binned distributions on the same axis."""
    p = np.asarray(reference_shares, dtype=float)
    q = np.asarray(current_shares, dtype=float)
    if p.shape != q.shape:
        raise ValueError("both distributions must have the same number of bins")
    if (p <= 0).any() or (q <= 0).any():
        raise ValueError("shares must be strictly positive; floor them with an epsilon first")
    return float(np.sum((q - p) * np.log(q / p)))


def binned_wasserstein(reference_shares, current_shares, points) -> float:
    """1-Wasserstein distance between two discrete distributions on a shared support."""
    p = np.asarray(reference_shares, dtype=float)
    q = np.asarray(current_shares, dtype=float)
    grid = np.asarray(points, dtype=float)
    if not (p.shape == q.shape == grid.shape):
        raise ValueError("shares and support points must have the same shape")
    if grid.size < 2:
        return 0.0
    difference = np.cumsum(p / p.sum()) - np.cumsum(q / q.sum())
    return float(np.sum(np.abs(difference[:-1]) * np.diff(grid)))


def bin_representatives(reference: NumericBaseline) -> np.ndarray:
    """Representative value per bin, with the open tails closed at the reference p01/p99."""
    edges = reference.edge_array().copy()
    if edges.size < 2:
        raise ValueError("the reference has no bins")
    edges[0] = min(reference.p01, edges[1])
    edges[-1] = max(reference.p99, edges[-2])
    return (edges[:-1] + edges[1:]) / 2.0


@dataclass(frozen=True)
class DriftPolicy:
    """Every threshold in one place, so an on-call engineer can tune it without reading code."""

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    tvd_warn: float = 0.10
    tvd_alert: float = 0.20
    unseen_warn: float = 0.01
    alpha: float = 0.05
    require_significance: bool = True
    noise_multiple: float = 3.0
    min_effective_rows: int = 50
    min_expected: float = 5.0
    epsilon: float = 1e-4

    def validate(self) -> None:
        if not 0 < self.psi_warn < self.psi_alert:
            raise ValueError("psi thresholds must satisfy 0 < warn < alert")
        if not 0 < self.tvd_warn < self.tvd_alert <= 1:
            raise ValueError("tvd thresholds must satisfy 0 < warn < alert <= 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if self.noise_multiple < 1:
            raise ValueError("noise_multiple below 1 would alert inside the noise floor")
        if self.min_effective_rows < 10:
            raise ValueError("min_effective_rows below 10 cannot support any test")


@dataclass(frozen=True)
class FeatureDrift:
    """One feature, one window."""

    feature: str
    kind: str
    rows: int
    effect_name: str
    effect: float
    warn_threshold: float
    alert_threshold: float
    noise_floor: float
    statistic: float
    p_value: float | None
    q_value: float | None
    severity: str
    note: str
    detail: dict = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.severity in ("warn", "alert")

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "rows": self.rows,
            "effect_name": self.effect_name,
            "effect": None if np.isnan(self.effect) else round(self.effect, 5),
            "warn_threshold": None if np.isnan(self.warn_threshold) else round(self.warn_threshold, 5),
            "alert_threshold": None if np.isnan(self.alert_threshold) else round(self.alert_threshold, 5),
            "noise_floor": None if np.isnan(self.noise_floor) else round(self.noise_floor, 5),
            "statistic": None if np.isnan(self.statistic) else round(self.statistic, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "q_value": None if self.q_value is None else round(self.q_value, 6),
            "severity": self.severity,
            "note": self.note,
            "detail": {
                key: (None if isinstance(value, float) and np.isnan(value) else value)
                for key, value in self.detail.items()
            },
        }


@dataclass(frozen=True)
class DriftReport:
    window: int
    rows: int
    features: tuple[FeatureDrift, ...]
    schema: SchemaReport
    policy: DriftPolicy

    @property
    def flagged(self) -> tuple[FeatureDrift, ...]:
        return tuple(item for item in self.features if item.flagged)

    @property
    def alerts(self) -> tuple[FeatureDrift, ...]:
        return tuple(item for item in self.features if item.severity == "alert")

    @property
    def verdict(self) -> str:
        if self.alerts or self.schema.blocking:
            return "action_required"
        if self.flagged or any(issue.severity == "warning" for issue in self.schema.issues):
            return "investigate"
        return "stable"

    def frame(self) -> pd.DataFrame:
        """Worst first: alerts, then warnings, then whatever could not be graded."""
        columns = [
            "feature",
            "kind",
            "rows",
            "effect_name",
            "effect",
            "warn_at",
            "noise_floor",
            "q_value",
            "severity",
        ]
        if not self.features:
            return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
        frame = pd.DataFrame(
            [
                {
                    "feature": item.feature,
                    "kind": item.kind,
                    "rows": item.rows,
                    "effect_name": item.effect_name,
                    "effect": float(item.effect),
                    "warn_at": float(item.warn_threshold),
                    "noise_floor": float(item.noise_floor),
                    "q_value": float("nan") if item.q_value is None else float(item.q_value),
                    "severity": item.severity,
                }
                for item in self.features
            ],
            columns=columns,
        )
        # an explicit rank column rather than a sort key: when every feature was skipped the
        # effect column is all-NaN, and a key that negates it would raise on an object dtype
        frame["_rank"] = frame["severity"].map(SEVERITY_RANK).fillna(9).astype(int)
        return (
            frame.sort_values(
                ["_rank", "effect"],
                ascending=[True, False],
                na_position="last",
                ignore_index=True,
            )
            .drop(columns="_rank")
            .round(
                {"effect": 4, "warn_at": 4, "noise_floor": 4, "q_value": 5}
            )
        )

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "rows": self.rows,
            "verdict": self.verdict,
            "flagged": [item.feature for item in self.flagged],
            "features": [item.to_dict() for item in self.features],
            "schema": self.schema.to_dict(),
        }


def _grade(
    effect: float,
    warn: float,
    alert: float,
    significant: bool | None,
    require_significance: bool,
) -> tuple[str, str]:
    """Severity plus the sentence explaining it."""
    if effect >= alert:
        level = "alert"
    elif effect >= warn:
        level = "warn"
    else:
        return "none", f"effect {effect:.4f} is below the {warn:.4f} threshold for this window"

    if require_significance and significant is False:
        demoted = "warn" if level == "alert" else "none"
        return demoted, (
            f"effect {effect:.4f} clears the {level} threshold but is not statistically "
            "supported after FDR correction at this window size, so it is reported as "
            f"{demoted!r} rather than paged"
        )
    return level, f"effect {effect:.4f} clears the {level} threshold and is significant"


def _numeric_drift(
    name: str,
    values,
    reference: NumericBaseline,
    policy: DriftPolicy,
    kind: str = "numeric",
) -> tuple[FeatureDrift, float | None]:
    coerced = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    clean = clean_numeric(coerced)
    edges = reference.edge_array()
    reference_shares = reference.share_array()
    n_bins = reference_shares.size

    if clean.size < policy.min_effective_rows:
        return (
            FeatureDrift(
                feature=name,
                kind=kind,
                rows=int(clean.size),
                effect_name="psi",
                effect=float("nan"),
                warn_threshold=float("nan"),
                alert_threshold=float("nan"),
                noise_floor=float("nan"),
                statistic=float("nan"),
                p_value=None,
                q_value=None,
                severity="insufficient_data",
                note=(
                    f"only {clean.size} usable values against a {policy.min_effective_rows}-row "
                    "minimum: any statistic here would be noise"
                ),
            ),
            None,
        )

    current_shares = bin_shares(clean, edges, policy.epsilon)
    psi_value = psi_from_shares(reference_shares, current_shares)
    noise = psi_noise_floor(reference.count, int(clean.size), n_bins)
    warn = max(policy.psi_warn, policy.noise_multiple * noise)
    alert = max(policy.psi_alert, 2.0 * policy.noise_multiple * noise)

    statistic, p_value, dof, pooled = chi_square_homogeneity(
        reference.reference_counts(), bin_counts(clean, edges), policy.min_expected
    )
    spread = max(reference.spread, 1e-12)
    detail = {
        "mean_shift_in_iqr": round(float((clean.mean() - reference.mean) / spread), 4),
        "wasserstein_in_iqr": round(
            float(
                binned_wasserstein(reference_shares, current_shares, bin_representatives(reference))
                / spread
            ),
            4,
        ),
        "ks_binned": round(
            float(np.max(np.abs(np.cumsum(reference_shares) - np.cumsum(current_shares)))), 4
        ),
        "current_mean": round(float(clean.mean()), 4),
        "reference_mean": round(float(reference.mean), 4),
        "chi2_dof": dof,
        "pooled_bins": pooled,
    }
    return (
        FeatureDrift(
            feature=name,
            kind=kind,
            rows=int(clean.size),
            effect_name="psi",
            effect=psi_value,
            warn_threshold=warn,
            alert_threshold=alert,
            noise_floor=noise,
            statistic=statistic,
            p_value=p_value,
            q_value=None,
            severity="none",
            note="",
            detail=detail,
        ),
        p_value,
    )


def _categorical_drift(
    name: str, values, reference: CategoricalBaseline, policy: DriftPolicy
) -> tuple[FeatureDrift, float | None]:
    counts = reference.encode(values)
    total = int(counts.sum())
    reference_counts = reference.reference_counts()

    if total < policy.min_effective_rows:
        return (
            FeatureDrift(
                feature=name,
                kind="categorical",
                rows=total,
                effect_name="tvd",
                effect=float("nan"),
                warn_threshold=float("nan"),
                alert_threshold=float("nan"),
                noise_floor=float("nan"),
                statistic=float("nan"),
                p_value=None,
                q_value=None,
                severity="insufficient_data",
                note=f"only {total} rows against a {policy.min_effective_rows}-row minimum",
            ),
            None,
        )

    current_shares = counts / total
    reference_shares = reference_counts / reference_counts.sum()
    tvd = total_variation(reference_shares, current_shares)
    noise = tvd_noise_floor(reference_shares, reference.count, total)
    warn = max(policy.tvd_warn, policy.noise_multiple * noise)
    alert = max(policy.tvd_alert, 2.0 * policy.noise_multiple * noise)

    statistic, p_value, dof, pooled = chi_square_homogeneity(
        reference_counts, counts, policy.min_expected
    )
    new_mass = unseen_share(counts)
    detail = {
        "jensen_shannon_bits": round(jensen_shannon(reference_shares, current_shares), 4),
        "cramers_v": round(cramers_v(statistic, total + int(reference.count), dof), 4),
        "unseen_share": round(new_mass, 4),
        "levels": len(reference.categories),
        "chi2_dof": dof,
        "pooled_levels": pooled,
    }
    return (
        FeatureDrift(
            feature=name,
            kind="categorical",
            rows=total,
            effect_name="tvd",
            effect=tvd,
            warn_threshold=warn,
            alert_threshold=alert,
            noise_floor=noise,
            statistic=statistic,
            p_value=p_value,
            q_value=None,
            severity="none",
            note="",
            detail=detail,
        ),
        p_value,
    )


def _skipped(name: str, kind: str, reason: str) -> FeatureDrift:
    return FeatureDrift(
        feature=name,
        kind=kind,
        rows=0,
        effect_name="tvd" if kind == "categorical" else "psi",
        effect=float("nan"),
        warn_threshold=float("nan"),
        alert_threshold=float("nan"),
        noise_floor=float("nan"),
        statistic=float("nan"),
        p_value=None,
        q_value=None,
        severity="skipped",
        note=reason,
    )


def scan_window(
    frame: pd.DataFrame,
    baseline: Baseline,
    policy: DriftPolicy | None = None,
    window: int = 0,
    schema: SchemaReport | None = None,
    prediction_column: str | None = None,
) -> DriftReport:
    """Compare one window against the baseline and grade every feature."""
    policy = policy or DriftPolicy()
    policy.validate()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    report = schema if schema is not None else check_schema(frame, baseline)

    window_blocked = any(issue.feature == WINDOW_SCOPE for issue in report.blocking)
    blocked = report.blocked_features

    drafts: list[FeatureDrift] = []
    pvalues: list[float] = []
    positions: list[int] = []

    def consider(name: str, kind: str, compute) -> None:
        if window_blocked:
            drafts.append(_skipped(name, kind, "the window itself failed validation"))
            return
        if name in blocked or name not in frame.columns:
            drafts.append(
                _skipped(
                    name,
                    kind,
                    "a blocking schema issue makes any distribution statistic on this column "
                    "meaningless this window",
                )
            )
            return
        drift, p_value = compute()
        if p_value is not None:
            positions.append(len(drafts))
            pvalues.append(p_value)
        drafts.append(drift)

    for name, reference in baseline.numeric.items():
        consider(
            name,
            "numeric",
            lambda name=name, reference=reference: _numeric_drift(
                name, frame[name], reference, policy
            ),
        )
    for name, reference in baseline.categorical.items():
        consider(
            name,
            "categorical",
            lambda name=name, reference=reference: _categorical_drift(
                name, frame[name], reference, policy
            ),
        )
    if prediction_column and baseline.prediction is not None:
        consider(
            prediction_column,
            "prediction",
            lambda: _numeric_drift(
                prediction_column,
                frame[prediction_column],
                baseline.prediction,
                policy,
                kind="prediction",
            ),
        )

    graded: list[FeatureDrift] = list(drafts)
    if pvalues:
        rejected, qvalues = benjamini_hochberg(np.asarray(pvalues), policy.alpha)
        for offset, index in enumerate(positions):
            draft = drafts[index]
            severity, note = _grade(
                draft.effect,
                draft.warn_threshold,
                draft.alert_threshold,
                bool(rejected[offset]),
                policy.require_significance,
            )
            unseen = float(draft.detail.get("unseen_share", 0.0))
            if draft.kind == "categorical" and unseen > policy.unseen_warn:
                # a level the model has never seen is a coding problem, not a distribution
                # shift, and it does not need statistical support to be worth a look
                if severity == "none":
                    severity = "warn"
                note += (
                    f"; {unseen:.2%} of rows fall in levels the reference never saw, which is "
                    "escalated on its own"
                )
            graded[index] = FeatureDrift(
                **{
                    **draft.__dict__,
                    "q_value": float(qvalues[offset]),
                    "severity": severity,
                    "note": note,
                }
            )

    return DriftReport(
        window=window,
        rows=len(frame),
        features=tuple(graded),
        schema=report,
        policy=policy,
    )


def decision_table() -> pd.DataFrame:
    """The four cases a monitoring report actually has to distinguish.

    This is printed by the CLI because it is the part that decides whether monitoring is useful:
    a feature drift alert on its own is not a reason to retrain, and a quality drop with stable
    inputs is not a data problem.
    """
    return pd.DataFrame(
        [
            {
                "feature drift": "no",
                "quality drop": "no",
                "reading": "stable",
                "action": "nothing; keep the baseline pinned",
            },
            {
                "feature drift": "yes",
                "quality drop": "no",
                "reading": "covariate shift the model absorbed",
                "action": "note it, do not retrain on a moved population without a reason",
            },
            {
                "feature drift": "no",
                "quality drop": "yes",
                "reading": "concept drift: the relationship changed, not the inputs",
                "action": "retrain on recent labelled data; feature monitoring cannot see this",
            },
            {
                "feature drift": "yes",
                "quality drop": "yes",
                "reading": "population moved and the model followed it down",
                "action": "investigate the pipeline first, then retrain",
            },
        ]
    )
