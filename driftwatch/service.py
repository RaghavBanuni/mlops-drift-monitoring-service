"""HTTP surface: a monitoring sidecar the scoring service can post batches to.

The API is deliberately small and the state is deliberately in-process.  A production
deployment would put the baseline in object storage and the window history in a database; the
boundary that matters for this repository is that :class:`driftwatch.monitor.Monitor` holds no
HTTP concepts and this module holds no statistics, so the maths is testable without a client and
replacing the storage never touches the detection code.

A missing baseline returns 409 rather than 200-with-nulls: a monitoring endpoint that answers
cheerfully while monitoring nothing is the failure mode this whole repo is arguing against.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .baseline import Baseline, fit_baseline
from .detect import decision_table
from .monitor import Monitor, MonitorConfig


class FitRequest(BaseModel):
    rows: list[dict] = Field(..., min_length=1)
    features: list[str] | None = None
    prediction_column: str | None = "prediction"
    target_column: str | None = "label"
    id_column: str | None = None
    segment_column: str | None = None
    n_bins: int = 10
    min_rows: int = 200
    model_version: str = "unversioned"


class LoadRequest(BaseModel):
    path: str
    prediction_column: str | None = "prediction"
    id_column: str | None = None
    segment_column: str | None = None


class WindowRequest(BaseModel):
    rows: list[dict] = Field(..., min_length=1)
    window: int | None = None


class LabelRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1)
    labels: list[float] = Field(..., min_length=1)
    window: int | None = None


def json_safe(value):
    """Make a payload valid JSON: NaN and infinity become null.

    Every statistic here has an honest "not measurable" state - an AUC with no matured labels, a
    PSI on a column that failed validation, a median label lag before any label arrived - and
    those are NaN internally.  ``NaN`` is not JSON, though, and emitting it produces a document
    that strict parsers reject; ``null`` says the same thing in a form every client can read.
    Numpy scalars are unwrapped here too, for the same reason.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def create_app(monitor: Monitor | None = None) -> FastAPI:
    """Build the app.  Injecting a monitor keeps the tests free of fitting overhead."""
    app = FastAPI(
        title="driftwatch",
        version="0.1.0",
        description="Drift and quality monitoring for a model already in production.",
    )
    state: dict[str, Monitor | None] = {"monitor": monitor}

    def current() -> Monitor:
        active = state["monitor"]
        if active is None:
            raise HTTPException(
                status_code=409,
                detail="no baseline is loaded; POST /baseline/fit or /baseline/load first",
            )
        return active

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "baseline_loaded": state["monitor"] is not None}

    @app.get("/decisions")
    def decisions() -> dict:
        """The drift-versus-quality decision table, served so it cannot be forgotten."""
        return {"cases": decision_table().to_dict(orient="records")}

    @app.post("/baseline/fit", status_code=201)
    def fit(request: FitRequest) -> dict:
        frame = pd.DataFrame(request.rows)
        try:
            baseline = fit_baseline(
                frame,
                features=request.features,
                prediction_column=request.prediction_column,
                target_column=request.target_column,
                n_bins=request.n_bins,
                min_rows=request.min_rows,
                model_version=request.model_version,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        state["monitor"] = Monitor(
            baseline,
            config=MonitorConfig(
                prediction_column=request.prediction_column,
                id_column=request.id_column,
                segment_column=request.segment_column,
            ),
        )
        return json_safe(state["monitor"].summary())

    @app.post("/baseline/load", status_code=201)
    def load(request: LoadRequest) -> dict:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no baseline file at {path}")
        state["monitor"] = Monitor(
            Baseline.load(path),
            config=MonitorConfig(
                prediction_column=request.prediction_column,
                id_column=request.id_column,
                segment_column=request.segment_column,
            ),
        )
        return json_safe(state["monitor"].summary())

    @app.get("/baseline")
    def read_baseline() -> dict:
        return json_safe(current().baseline.to_dict())

    @app.post("/windows")
    def ingest(request: WindowRequest) -> dict:
        monitor = current()
        try:
            return json_safe(monitor.ingest(pd.DataFrame(request.rows), request.window))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/windows/{window}")
    def read_window(window: int) -> dict:
        monitor = current()
        try:
            return json_safe(monitor._report(window).to_dict())
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/labels")
    def labels(request: LabelRequest) -> dict:
        if len(request.ids) != len(request.labels):
            raise HTTPException(status_code=422, detail="ids and labels must be the same length")
        return json_safe(current().add_labels(request.ids, request.labels, request.window))

    @app.get("/quality")
    def quality(last_windows: int | None = None) -> dict:
        monitor = current()
        try:
            return json_safe(monitor.quality(last_windows=last_windows).to_dict())
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/interpretation")
    def interpretation() -> dict:
        return json_safe(current().interpret())

    @app.get("/alerts")
    def alerts() -> dict:
        monitor = current()
        return json_safe(
            {
                "stats": monitor.alerts.stats(),
                "alerts": [alert.to_dict() for alert in monitor.alerts.raised],
            }
        )

    @app.get("/history")
    def history() -> dict:
        return json_safe({"windows": current().history().to_dict(orient="records")})

    @app.get("/summary")
    def summary() -> dict:
        return json_safe(current().summary())

    return app


app = create_app()
