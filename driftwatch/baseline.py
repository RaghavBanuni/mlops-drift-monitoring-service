"""The reference the model was trained against, pinned and serialisable.

A baseline is part of the model artefact, not a rolling view of recent traffic.  Comparing this
week against last week - the default in most dashboards - makes *gradual* drift invisible: the
reference walks along with the data, every step looks small, and a year later the model is
scoring a population it never saw while every check stayed green.  So the reference here is
fitted once, versioned with the model, and saved to JSON alongside it.

What gets stored is deliberately small: quantile bin edges, the reference share in each bin,
category shares, null rates and summary statistics.  No raw reference rows, which keeps the
artefact shareable and free of subject data, and makes the comparison reproducible - the edges
cannot move because they are literally written down.  The cost of that choice is that every
reference-side statistic is computed on the binned distribution; :mod:`driftwatch.detect` says
so where it matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    NULL_LABEL,
    bin_shares,
    categorical_counts,
    clean_numeric,
    null_rate,
    quantile_bin_edges,
    reference_spread,
)
from .performance import roc_auc

OTHER_LABEL = "__other__"


@dataclass(frozen=True)
class NumericBaseline:
    """Fixed quantile edges, the reference mass in each bin, and summary statistics.

    ``shares`` is stored rather than assumed to be uniform.  Quantile edges collapse on ties, so
    a feature that is 40% zeros ends up with bins of very unequal mass; assuming 1/k per bin
    would make PSI report drift on the reference itself.
    """

    name: str
    count: int
    edges: tuple[float, ...]
    shares: tuple[float, ...]
    mean: float
    std: float
    spread: float
    minimum: float
    maximum: float
    p01: float
    p99: float
    null_rate: float

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        # JSON has no infinity; None round-trips to the open tail
        payload["edges"] = [None if np.isinf(edge) else float(edge) for edge in self.edges]
        payload["shares"] = [float(share) for share in self.shares]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "NumericBaseline":
        edges = tuple(
            (-np.inf if index == 0 else np.inf) if edge is None else float(edge)
            for index, edge in enumerate(payload["edges"])
        )
        return cls(
            **{
                **payload,
                "edges": edges,
                "shares": tuple(float(share) for share in payload["shares"]),
            }
        )

    def edge_array(self) -> np.ndarray:
        return np.asarray(self.edges, dtype=float)

    def share_array(self) -> np.ndarray:
        return np.asarray(self.shares, dtype=float)

    def reference_counts(self) -> np.ndarray:
        return self.share_array() * self.count


@dataclass(frozen=True)
class CategoricalBaseline:
    """Reference category shares, with the long tail folded into one bucket.

    ``rare`` records exactly which levels were folded, so a window containing one of them is
    mapped to :data:`OTHER_LABEL` instead of being reported as a brand-new category.  Getting
    this wrong produces a permanent stream of false "new level" alerts on any high-cardinality
    column.
    """

    name: str
    count: int
    categories: tuple[str, ...]
    shares: tuple[float, ...]
    rare: tuple[str, ...]
    null_rate: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "categories": list(self.categories),
            "shares": [float(share) for share in self.shares],
            "rare": list(self.rare),
            "null_rate": self.null_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CategoricalBaseline":
        return cls(
            name=payload["name"],
            count=int(payload["count"]),
            categories=tuple(payload["categories"]),
            shares=tuple(float(share) for share in payload["shares"]),
            rare=tuple(payload.get("rare", ())),
            null_rate=float(payload["null_rate"]),
        )

    def normalise(self, values) -> pd.Series:
        """Map a raw column onto the reference vocabulary (nulls and folded levels included)."""
        series = pd.Series(list(values), dtype="object")
        keys = series.where(series.notna(), NULL_LABEL).astype(str)
        if self.rare:
            keys = keys.where(~keys.isin(self.rare), OTHER_LABEL)
        return keys

    def encode(self, values) -> np.ndarray:
        """Counts aligned to :attr:`categories`, with a trailing slot for unseen levels."""
        return categorical_counts(self.normalise(values), list(self.categories))

    def reference_counts(self) -> np.ndarray:
        """Expected counts on the same axis, with zero mass in the unseen slot."""
        return np.append(np.asarray(self.shares, dtype=float) * self.count, 0.0)


@dataclass(frozen=True)
class Baseline:
    """Everything a window is compared against."""

    numeric: dict[str, NumericBaseline]
    categorical: dict[str, CategoricalBaseline]
    n_rows: int
    n_bins: int
    created_at: str
    model_version: str = "unversioned"
    prediction: NumericBaseline | None = None
    baseline_auc: float | None = None
    positive_rate: float | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    def kind_of(self, feature: str) -> str:
        if feature in self.numeric:
            return "numeric"
        if feature in self.categorical:
            return "categorical"
        raise KeyError(f"{feature!r} is not part of this baseline")

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "model_version": self.model_version,
            "n_rows": self.n_rows,
            "n_bins": self.n_bins,
            "baseline_auc": self.baseline_auc,
            "positive_rate": self.positive_rate,
            "numeric": {name: value.to_dict() for name, value in self.numeric.items()},
            "categorical": {name: value.to_dict() for name, value in self.categorical.items()},
            "prediction": None if self.prediction is None else self.prediction.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Baseline":
        return cls(
            numeric={
                name: NumericBaseline.from_dict(value)
                for name, value in payload.get("numeric", {}).items()
            },
            categorical={
                name: CategoricalBaseline.from_dict(value)
                for name, value in payload.get("categorical", {}).items()
            },
            n_rows=int(payload["n_rows"]),
            n_bins=int(payload["n_bins"]),
            created_at=payload["created_at"],
            model_version=payload.get("model_version", "unversioned"),
            prediction=(
                None
                if payload.get("prediction") is None
                else NumericBaseline.from_dict(payload["prediction"])
            ),
            baseline_auc=payload.get("baseline_auc"),
            positive_rate=payload.get("positive_rate"),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _fit_numeric(name: str, values, n_bins: int, epsilon: float) -> NumericBaseline:
    clean = clean_numeric(values)
    if clean.size < 2:
        raise ValueError(f"feature {name!r} has fewer than two finite reference values")
    edges = quantile_bin_edges(clean, n_bins)
    return NumericBaseline(
        name=name,
        count=int(clean.size),
        edges=tuple(float(edge) for edge in edges),
        shares=tuple(float(share) for share in bin_shares(clean, edges, epsilon)),
        mean=float(clean.mean()),
        std=float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        spread=float(reference_spread(clean)),
        minimum=float(clean.min()),
        maximum=float(clean.max()),
        p01=float(np.quantile(clean, 0.01)),
        p99=float(np.quantile(clean, 0.99)),
        null_rate=round(null_rate(values), 5),
    )


def _fit_categorical(
    name: str, values, max_categories: int, min_share: float
) -> CategoricalBaseline:
    series = pd.Series(list(values), dtype="object")
    keys = series.where(series.notna(), NULL_LABEL).astype(str)
    counts = keys.value_counts()
    if counts.empty:
        raise ValueError(f"feature {name!r} has no values in the reference")
    shares = counts / counts.sum()

    kept = shares[shares >= min_share].head(max_categories)
    if kept.empty:  # everything is rare: keep the most frequent level so the axis is not empty
        kept = shares.head(1)
    rare = tuple(sorted(set(shares.index) - set(kept.index)))

    categories = list(kept.index)
    category_shares = [float(share) for share in kept.to_numpy()]
    if rare:
        categories.append(OTHER_LABEL)
        category_shares.append(float(shares[list(rare)].sum()))

    total = float(sum(category_shares))
    return CategoricalBaseline(
        name=name,
        count=int(len(keys)),
        categories=tuple(categories),
        shares=tuple(share / total for share in category_shares),
        rare=rare,
        null_rate=round(float((keys == NULL_LABEL).mean()), 5),
    )


def infer_kinds(
    frame: pd.DataFrame, columns: list[str], max_numeric_levels: int = 15
) -> dict[str, str]:
    """Split columns into numeric and categorical.

    A numeric column with very few distinct values is treated as categorical: postcode area,
    product tier and "number of children" are codes, and quantile-binning a five-level integer
    produces collapsed edges and a meaningless PSI.
    """
    kinds: dict[str, str] = {}
    for column in columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            kinds[column] = "categorical"
        elif series.nunique(dropna=True) <= max_numeric_levels:
            kinds[column] = "categorical"
        else:
            kinds[column] = "numeric"
    return kinds


def fit_baseline(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    prediction_column: str | None = None,
    target_column: str | None = None,
    n_bins: int = 10,
    max_categories: int = 30,
    min_category_share: float = 0.005,
    min_rows: int = 200,
    epsilon: float = 1e-4,
    model_version: str = "unversioned",
) -> Baseline:
    """Fit a reference from a stable, representative period.

    ``min_rows`` is enforced because a thin reference is worse than no reference: the noise floor
    of PSI is roughly ``(bins - 1) * (1/n_ref + 1/n_cur)``, so a 200-row reference with ten bins
    already produces PSI near 0.05 on identically distributed data - half of the conventional
    "investigate" threshold, before anything has drifted at all.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if len(frame) < min_rows:
        raise ValueError(
            f"the reference has {len(frame)} rows; at least {min_rows} are needed for the "
            "comparison to mean anything"
        )

    reserved = {name for name in (prediction_column, target_column) if name}
    if features is None and numeric is None and categorical is None:
        features = [column for column in frame.columns if column not in reserved]
    if features is not None:
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise ValueError(f"columns not present in the reference frame: {missing}")
        kinds = infer_kinds(frame, [column for column in features if column not in reserved])
        numeric = [name for name, kind in kinds.items() if kind == "numeric"]
        categorical = [name for name, kind in kinds.items() if kind == "categorical"]
    numeric = list(numeric or [])
    categorical = list(categorical or [])
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(f"columns declared both numeric and categorical: {sorted(overlap)}")
    if not numeric and not categorical:
        raise ValueError("a baseline needs at least one feature")

    prediction = (
        _fit_numeric(prediction_column, frame[prediction_column], n_bins, epsilon)
        if prediction_column
        else None
    )

    baseline_auc = None
    positive_rate = None
    if target_column and prediction_column:
        labels = frame[target_column].to_numpy(dtype=float)
        scores = frame[prediction_column].to_numpy(dtype=float)
        mask = np.isfinite(labels) & np.isfinite(scores)
        if mask.sum() >= 2 and len(np.unique(labels[mask])) == 2:
            baseline_auc = round(float(roc_auc(labels[mask], scores[mask])), 5)
            positive_rate = round(float(labels[mask].mean()), 5)

    return Baseline(
        numeric={name: _fit_numeric(name, frame[name], n_bins, epsilon) for name in numeric},
        categorical={
            name: _fit_categorical(name, frame[name], max_categories, min_category_share)
            for name in categorical
        },
        n_rows=int(len(frame)),
        n_bins=n_bins,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_version=model_version,
        prediction=prediction,
        baseline_auc=baseline_auc,
        positive_rate=positive_rate,
    )
