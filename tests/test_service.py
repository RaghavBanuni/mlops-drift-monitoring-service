"""The HTTP surface, exercised through a client rather than by calling the functions.

fastapi and httpx are optional extras, so the whole module skips cleanly without them - a
monitoring library should not force a web stack on a batch user.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from driftwatch.monitor import Monitor, MonitorConfig  # noqa: E402
from driftwatch.service import create_app, json_safe  # noqa: E402
from driftwatch.simulate import FEATURES  # noqa: E402


def records(frame) -> list[dict]:
    """Rows as a client would send them: JSON types only, NaN as null."""
    return json.loads(frame.to_json(orient="records"))


@pytest.fixture
def client(baseline):
    monitor = Monitor(
        baseline,
        config=MonitorConfig(prediction_column="prediction", id_column="row_id"),
    )
    return TestClient(create_app(monitor))


@pytest.fixture
def bare():
    return TestClient(create_app())


class TestJsonSafety:
    def test_non_finite_numbers_become_null(self):
        payload = json_safe({"auc": float("nan"), "edge": float("inf"), "ok": 0.5})
        assert payload == {"auc": None, "edge": None, "ok": 0.5}

    def test_nested_structures_are_walked(self):
        payload = json_safe({"a": [{"b": float("-inf")}], "c": (1.5, "text")})
        assert payload == {"a": [{"b": None}], "c": [1.5, "text"]}

    def test_the_result_is_strict_json(self):
        text = json.dumps(json_safe({"auc": float("nan")}), allow_nan=False)
        assert text == '{"auc": null}'


class TestWithoutABaseline:
    def test_health_is_honest_about_monitoring_nothing(self, bare):
        response = bare.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "baseline_loaded": False}

    def test_every_monitoring_route_refuses_rather_than_inventing_an_answer(self, bare, stable):
        assert bare.get("/summary").status_code == 409
        assert bare.get("/baseline").status_code == 409
        assert bare.get("/quality").status_code == 409
        assert bare.get("/interpretation").status_code == 409
        assert bare.get("/history").status_code == 409
        assert bare.post("/windows", json={"rows": records(stable.windows[0].head(5))}).status_code == 409

    def test_the_decision_table_needs_no_baseline(self, bare):
        response = bare.get("/decisions")
        assert response.status_code == 200
        cases = response.json()["cases"]
        assert len(cases) == 4
        assert {case["reading"] for case in cases} >= {"stable"}


class TestBaselineRoutes:
    def test_the_stored_baseline_is_valid_json_with_no_infinities(self, client):
        response = client.get("/baseline")
        assert response.status_code == 200
        payload = response.json()
        assert set(payload["numeric"]) >= {"income", "credit_score"}
        assert payload["numeric"]["income"]["edges"][0] is None
        assert payload["baseline_auc"] is not None

    def test_fitting_over_http_replaces_the_baseline(self, bare, stable):
        response = bare.post(
            "/baseline/fit",
            json={
                "rows": records(stable.reference.head(400)),
                "features": list(FEATURES),
                "min_rows": 300,
                "model_version": "http-1",
                "id_column": "row_id",
            },
        )
        assert response.status_code == 201
        assert response.json()["model_version"] == "http-1"
        assert bare.get("/health").json()["baseline_loaded"] is True

    def test_a_reference_too_thin_to_mean_anything_is_rejected(self, bare, stable):
        response = bare.post(
            "/baseline/fit",
            json={"rows": records(stable.reference.head(50)), "features": list(FEATURES)},
        )
        assert response.status_code == 422
        assert "rows" in response.json()["detail"]

    def test_an_empty_body_is_rejected_by_the_schema(self, bare):
        assert bare.post("/baseline/fit", json={"rows": []}).status_code == 422

    def test_a_saved_baseline_can_be_loaded_from_disk(self, bare, baseline, tmp_path):
        path = baseline.save(tmp_path / "baseline.json")
        response = bare.post("/baseline/load", json={"path": str(path)})
        assert response.status_code == 201
        assert response.json()["model_version"] == "test-1"

    def test_a_missing_baseline_file_is_a_404(self, bare, tmp_path):
        response = bare.post("/baseline/load", json={"path": str(tmp_path / "absent.json")})
        assert response.status_code == 404


class TestMonitoringFlow:
    def test_a_window_is_scored_and_retrievable(self, client, stable):
        posted = client.post("/windows", json={"rows": records(stable.windows[0]), "window": 0})
        assert posted.status_code == 200
        body = posted.json()
        assert body["window"] == 0
        assert body["rows"] == len(stable.windows[0])
        assert body["verdict"] in ("stable", "investigate", "action_required")
        assert body["schema_blocking"] == []

        stored = client.get("/windows/0")
        assert stored.status_code == 200
        assert len(stored.json()["features"]) == len(FEATURES) + 1  # plus the prediction column

    def test_an_unscored_window_is_a_404(self, client):
        assert client.get("/windows/99").status_code == 404

    def test_a_broken_window_is_reported_not_hidden(self, client, stable):
        window = stable.windows[0].copy()
        window["income"] = "n/a"
        body = client.post("/windows", json={"rows": records(window), "window": 1}).json()

        assert body["verdict"] == "action_required"
        assert [issue["feature"] for issue in body["schema_blocking"]] == ["income"]
        assert any(alert["kind"].startswith("schema") for alert in body["alerts"])

    def test_outcomes_arrive_separately_and_unlock_quality(self, client, stable):
        window = stable.windows[0]
        client.post("/windows", json={"rows": records(window), "window": 0})

        assert client.get("/quality").json()["verdict"] == "insufficient_labels"

        labels = client.post(
            "/labels",
            json={
                "ids": list(window["row_id"]),
                "labels": [float(value) for value in window["label"]],
                "window": 1,
            },
        )
        assert labels.status_code == 200
        assert labels.json()["matched"] == len(window)
        assert labels.json()["coverage"] == 1.0

        quality = client.get("/quality").json()
        assert quality["verdict"] in ("stable", "degraded", "improved")
        assert quality["auc"] is not None
        assert len(quality["auc_ci"]) == 2

    def test_mismatched_labels_are_rejected(self, client):
        response = client.post("/labels", json={"ids": ["a", "b"], "labels": [1.0]})
        assert response.status_code == 422

    def test_the_interpretation_reports_pending_outcomes_as_pending(self, client, stable):
        client.post("/windows", json={"rows": records(stable.windows[0]), "window": 0})
        reading = client.get("/interpretation").json()
        assert reading["case"].endswith("outcomes_pending")
        assert reading["quality"] is None

    def test_history_and_alerts_accumulate(self, client, stable):
        for index in range(3):
            client.post("/windows", json={"rows": records(stable.windows[index]), "window": index})

        history = client.get("/history").json()["windows"]
        assert [row["window"] for row in history] == [0, 1, 2]

        alerts = client.get("/alerts").json()
        assert alerts["stats"]["raised"] == len(alerts["alerts"])

        summary = client.get("/summary").json()
        assert summary["windows_seen"] == 3
        assert summary["features"] == len(FEATURES)
        assert summary["coverage"]["median_lag_windows"] is None  # nothing labelled yet
