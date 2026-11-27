"""FastAPI service and JSON logging tests using TestClient."""

import json
import math
from io import StringIO

import pytest
from fastapi.testclient import TestClient

from app.main import app, logger

VALID_PAYLOAD = {
    "cards_attached": 1200,
    "working_hours": 7.5,
    "biometric_failures": 5,
    "temp_c": 31.0,
    "stock_kg_morning": 1250.0,
    "days_to_month_end": 9,
    "shop_area": "rural",
    "block": "Block North",
}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_boundary_and_json_log(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    buffer = StringIO()
    monkeypatch.setattr(logger.handlers[0], "stream", buffer)

    payload = {**VALID_PAYLOAD, "cards_attached": 3500, "biometric_failures": 0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert math.isfinite(prediction)

    log_line = json.loads(buffer.getvalue().strip())
    assert log_line["message"] == "prediction_served"
    assert log_line["status_code"] == 200
    assert log_line["prediction"] == pytest.approx(round(prediction, 4))
    assert "latency_ms" in log_line


@pytest.mark.parametrize(
    "invalid_update",
    [
        {"cards_attached": 3501},
        {"biometric_failures": -1},
        {"record_id": "R99999"},
    ],
)
def test_predict_rejects_invalid_inputs_with_422(client: TestClient, invalid_update: dict):
    response = client.post("/predict", json={**VALID_PAYLOAD, **invalid_update})
    assert response.status_code == 422
