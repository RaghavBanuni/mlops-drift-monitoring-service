"""Synthetic production traffic with known change points - and a harness that scores the monitor.

Almost every drift-monitoring repository demonstrates that its detector fires on injected drift.
That is not a result: a detector with no threshold discipline fires on everything.  The
interesting question is the trade - how many windows late does it fire, and how often does it
fire when nothing happened - and answering it needs ground truth, which is why the data here is
generated.

The deployed model is fixed.  Its coefficients, its feature scaling and its median imputation
are all frozen at reference time, exactly as a deployed artefact would be.  Scenarios then move
the world around it:

``stable``             nothing changes; every alarm here is a false alarm.
``benign_seasonal``    the channel mix oscillates.  Real drift, no harm - the case that decides
                       whether a monitor is usable, because alerting on it teaches people to
                       ignore it.
``gradual_covariate``  income creeps upward window by window; each step is small.
``sudden_covariate``   income and utilisation jump at the change point.
``concept``            **the features do not move at all**; the label-generating relationship
                       changes.  No feature-drift detector can see this, which is the point.
``new_category``       a region level the reference never saw appears.
``null_spike``         credit score starts arriving null 35% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alerts import AlertManager, AlertPolicy
from .baseline import fit_baseline
from .detect import DriftPolicy, scan_window
from .metrics import ks_test
from .performance import roc_auc
from .sequential import PageHinkley
from .stats import benjamini_hochberg, bonferroni

FEATURES: tuple[str, ...] = (
    "income",
    "credit_score",
    "utilisation",
    "tenure_months",
    "region",
    "channel",
    "n_products",
)
NUMERIC_FEATURES: tuple[str, ...] = ("income", "credit_score", "utilisation", "tenure_months")
REGIONS: tuple[str, ...] = ("north", "south", "east", "west", "central")
CHANNELS: tuple[str, ...] = ("branch", "web", "broker")

SCENARIOS: tuple[str, ...] = (
    "stable",
    "benign_seasonal",
    "gradual_covariate",
    "sudden_covariate",
    "concept",
    "new_category",
    "null_spike",
)

# frozen at reference time: the deployed model never learns anything again
SCALING = {
    "income": (10.5, 0.45),
    "credit_score": (650.0, 60.0),
    "utilisation": (0.35, 0.18),
    "tenure_months": (36.0, 18.0),
}
MODEL_COEFFICIENTS = {
    "intercept": -1.6,
    "income": -0.5,
    "credit_score": -0.9,
    "utilisation": 0.8,
    "tenure_months": -0.3,
    "region_west": 0.35,
    "channel_broker": 0.3,
}
# the world's relationship after concept drift: utilisation stops mattering, tenure starts
DRIFTED_COEFFICIENTS = {
    **MODEL_COEFFICIENTS,
    "utilisation": -0.1,
    "credit_score": -0.35,
    "tenure_months": -0.9,
}


@dataclass(frozen=True)
class ScenarioConfig:
    n_reference: int = 4_000
    n_window: int = 800
    n_windows: int = 20
    change_at: int = 10
    seed: int = 7

    def validate(self) -> None:
        if self.n_reference < 500:
            raise ValueError("the reference needs at least 500 rows")
        if self.n_window < 100:
            raise ValueError("windows below 100 rows cannot support any test")
        if not 1 <= self.change_at < self.n_windows:
            raise ValueError("change_at must fall inside the window range")


@dataclass(frozen=True)
class Scenario:
    name: str
    reference: pd.DataFrame
    windows: tuple[pd.DataFrame, ...]
    change_at: int | None
    description: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw(
    n: int,
    rng: np.random.Generator,
    income_shift: float = 0.0,
    utilisation_shift: float = 0.0,
    web_share: float = 0.35,
    extra_region: tuple[str, float] | None = None,
    credit_null_rate: float = 0.0,
) -> pd.DataFrame:
    """One batch of applications.  Shifts are expressed in standard deviations."""
    z_income = rng.normal(income_shift, 1.0, n)
    z_score = rng.normal(0.0, 1.0, n)
    z_util = rng.normal(utilisation_shift, 1.0, n)
    z_tenure = rng.normal(0.0, 1.0, n)

    regions = list(REGIONS)
    weights = np.array([0.24, 0.22, 0.2, 0.18, 0.16])
    if extra_region is not None:
        name, share = extra_region
        regions = regions + [name]
        weights = np.append(weights * (1.0 - share), share)
    weights = weights / weights.sum()

    channel_weights = np.array([1.0 - web_share - 0.2, web_share, 0.2])
    channel_weights = np.clip(channel_weights, 0.05, None)
    channel_weights = channel_weights / channel_weights.sum()

    frame = pd.DataFrame(
        {
            "income": np.exp(SCALING["income"][0] + SCALING["income"][1] * z_income),
            "credit_score": np.clip(
                SCALING["credit_score"][0] + SCALING["credit_score"][1] * z_score, 300, 850
            ),
            "utilisation": np.clip(
                SCALING["utilisation"][0] + SCALING["utilisation"][1] * z_util, 0.0, 1.5
            ),
            "tenure_months": np.clip(
                np.round(SCALING["tenure_months"][0] + SCALING["tenure_months"][1] * z_tenure),
                0,
                240,
            ),
            "region": rng.choice(regions, size=n, p=weights),
            "channel": rng.choice(list(CHANNELS), size=n, p=channel_weights),
            "n_products": rng.integers(1, 5, n),
        }
    )
    if credit_null_rate > 0:
        missing = rng.random(n) < credit_null_rate
        frame.loc[missing, "credit_score"] = np.nan
    return frame


def _z_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recompute the model's inputs with the frozen scaling and frozen median imputation."""
    z: dict[str, np.ndarray] = {}
    for name, (centre, scale) in SCALING.items():
        values = frame[name].to_numpy(dtype=float)
        if name == "income":
            values = np.log(np.clip(values, 1.0, None))
        filled = np.where(np.isfinite(values), values, centre)  # frozen imputation
        z[name] = (filled - centre) / scale
    return z


def _logit(frame: pd.DataFrame, coefficients: dict[str, float]) -> np.ndarray:
    z = _z_scores(frame)
    total = np.full(len(frame), coefficients["intercept"], dtype=float)
    for name in SCALING:
        total = total + coefficients[name] * z[name]
    total = total + coefficients["region_west"] * (frame["region"].to_numpy() == "west")
    total = total + coefficients["channel_broker"] * (frame["channel"].to_numpy() == "broker")
    return total


def score(frame: pd.DataFrame) -> np.ndarray:
    """The deployed model.  Fixed forever, imputation included."""
    return _sigmoid(_logit(frame, MODEL_COEFFICIENTS))


def label(
    frame: pd.DataFrame, rng: np.random.Generator, coefficients: dict[str, float] | None = None
) -> np.ndarray:
    """Draw outcomes from the world's current relationship, which drift can change."""
    probability = _sigmoid(_logit(frame, coefficients or MODEL_COEFFICIENTS))
    return (rng.random(len(frame)) < probability).astype(int)


def _finalise(
    frame: pd.DataFrame, rng: np.random.Generator, coefficients: dict[str, float], offset: int
) -> pd.DataFrame:
    frame = frame.copy()
    frame["row_id"] = [f"r{offset + index}" for index in range(len(frame))]
    frame["prediction"] = score(frame)
    frame["label"] = label(frame, rng, coefficients)
    return frame


def build_scenario(name: str, config: ScenarioConfig | None = None) -> Scenario:
    """Reference plus a sequence of windows, with the change point recorded."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")
    settings = config or ScenarioConfig()
    settings.validate()
    rng = np.random.default_rng(settings.seed)

    reference = _finalise(_draw(settings.n_reference, rng), rng, MODEL_COEFFICIENTS, 0)
    windows: list[pd.DataFrame] = []
    change_at: int | None = settings.change_at
    if name in ("stable", "benign_seasonal"):
        change_at = None

    for index in range(settings.n_windows):
        after = index >= settings.change_at
        kwargs: dict = {}
        coefficients = MODEL_COEFFICIENTS
        if name == "benign_seasonal":
            kwargs["web_share"] = 0.35 + 0.12 * np.sin(index / 3.0)
        elif name == "gradual_covariate" and after:
            kwargs["income_shift"] = 0.06 * (index - settings.change_at + 1)
        elif name == "sudden_covariate" and after:
            kwargs["income_shift"] = 0.8
            kwargs["utilisation_shift"] = 0.5
        elif name == "concept" and after:
            coefficients = DRIFTED_COEFFICIENTS
        elif name == "new_category" and after:
            kwargs["extra_region"] = ("overseas", 0.12)
        elif name == "null_spike" and after:
            kwargs["credit_null_rate"] = 0.35

        batch = _draw(settings.n_window, rng, **kwargs)
        windows.append(
            _finalise(
                batch,
                rng,
                coefficients,
                settings.n_reference + index * settings.n_window,
            )
        )

    descriptions = {
        "stable": "nothing changes; every alarm is a false alarm",
        "benign_seasonal": "channel mix oscillates - real drift, no harm",
        "gradual_covariate": "income creeps up by 0.06 sd per window",
        "sudden_covariate": "income +0.8 sd and utilisation +0.5 sd at the change point",
        "concept": "features unchanged; the label relationship changes",
        "new_category": "a region level the reference never saw takes 12% of traffic",
        "null_spike": "credit score arrives null 35% of the time",
    }
    return Scenario(
        name=name,
        reference=reference,
        windows=tuple(windows),
        change_at=change_at,
        description=descriptions[name],
    )


def _ks_pvalues(reference: pd.DataFrame, window: pd.DataFrame) -> np.ndarray:
    return np.array(
        [ks_test(reference[name], window[name])[1] for name in NUMERIC_FEATURES], dtype=float
    )


def _summarise(name: str, alarms: list[bool], change_at: int | None) -> dict:
    """False alarms before the change, and how many windows late the first true alarm was."""
    if change_at is None:
        false_alarms = int(sum(alarms))
        return {
            "detector": name,
            "false_alarm_windows": false_alarms,
            "false_alarm_rate": round(false_alarms / max(len(alarms), 1), 3),
            "detected": None,
            "delay_windows": None,
        }
    before = alarms[:change_at]
    after = alarms[change_at:]
    detected = any(after)
    delay = int(np.argmax(after)) if detected else None
    return {
        "detector": name,
        "false_alarm_windows": int(sum(before)),
        "false_alarm_rate": round(sum(before) / max(len(before), 1), 3),
        "detected": detected,
        "delay_windows": delay,
    }


def evaluate_detectors(
    scenario: str,
    config: ScenarioConfig | None = None,
    policy: DriftPolicy | None = None,
    alert_policy: AlertPolicy | None = None,
) -> pd.DataFrame:
    """Score several detection rules on the same scenario.

    Read the table by column, not by row: a detector with zero delay and six false alarms is not
    better than one with a two-window delay and none.  Which trade is right depends on what an
    investigation costs, and that is a business decision the monitor cannot make.
    """
    built = build_scenario(scenario, config)
    policy = policy or DriftPolicy()
    baseline = fit_baseline(
        built.reference,
        features=list(FEATURES),
        prediction_column="prediction",
        target_column="label",
        min_rows=200,
    )
    reports = [
        scan_window(window, baseline, policy, window=index, prediction_column="prediction")
        for index, window in enumerate(built.windows)
    ]
    pvalue_sets = [_ks_pvalues(built.reference, window) for window in built.windows]

    rows = [
        _summarise(
            "KS per feature, alpha=0.05 (no correction)",
            [bool((pvalues <= 0.05).any()) for pvalues in pvalue_sets],
            built.change_at,
        ),
        _summarise(
            "KS + Bonferroni",
            [bool(bonferroni(pvalues, 0.05).any()) for pvalues in pvalue_sets],
            built.change_at,
        ),
        _summarise(
            "KS + Benjamini-Hochberg",
            [bool(benjamini_hochberg(pvalues, 0.05)[0].any()) for pvalues in pvalue_sets],
            built.change_at,
        ),
        _summarise(
            "PSI > 0.1, fixed threshold",
            [
                any(
                    item.kind == "numeric"
                    and not np.isnan(item.effect)
                    and item.effect > 0.1
                    for item in report.features
                )
                for report in reports
            ],
            built.change_at,
        ),
        _summarise(
            "driftwatch policy (effect size + adaptive floor + FDR)",
            [bool(report.flagged) for report in reports],
            built.change_at,
        ),
    ]

    manager = AlertManager(policy=alert_policy or AlertPolicy())
    confirmed = [
        any(alert.kind.startswith("drift") for alert in manager.process(report))
        for report in reports
    ]
    rows.append(
        _summarise("driftwatch policy + 2-of-3 confirmation", confirmed, built.change_at)
    )

    sigma = float(np.std([window["prediction"].mean() for window in built.windows[: max(built.change_at or 3, 3)]]))
    detector = PageHinkley(delta=0.25 * max(sigma, 1e-6), threshold=3.0 * max(sigma, 1e-6))
    rows.append(
        _summarise(
            "Page-Hinkley on the prediction mean",
            [detector.update(float(window["prediction"].mean()), index) for index, window in enumerate(built.windows)],
            built.change_at,
        )
    )

    quality = []
    baseline_auc = float(baseline.baseline_auc or 0.0)
    for window in built.windows:
        window_auc = roc_auc(window["label"], window["prediction"])
        quality.append(bool(np.isfinite(window_auc) and window_auc < baseline_auc - 0.03))
    rows.append(_summarise("labelled AUC drop > 3 points (needs outcomes)", quality, built.change_at))

    frame = pd.DataFrame(rows)
    frame.insert(0, "scenario", built.name)
    return frame


def evaluate_all(
    config: ScenarioConfig | None = None, scenarios: tuple[str, ...] = SCENARIOS
) -> pd.DataFrame:
    """Every scenario, one table.  This is the headline artefact of the repository."""
    return pd.concat(
        [evaluate_detectors(name, config) for name in scenarios], ignore_index=True
    )
