"""FastAPI prediction service for MP124 (Bhopal ration shop grain distribution)."""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.logging_config import get_json_logger  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"
logger = get_json_logger("mp124.service")


class GrainPredictionRequest(BaseModel):
    """Input schema enforcing the client's eight fields and two boundary rules."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cards_attached: int = Field(..., le=3500)
    working_hours: float
    biometric_failures: int = Field(..., ge=0)
    temp_c: float
    stock_kg_morning: float
    days_to_month_end: int
    shop_area: str
    block: str


class GrainPredictionResponse(BaseModel):
    """Predicted daily grain distribution in kilograms."""

    prediction: float


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="MP124 Ration Shop Distribution API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness and readiness probe endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=GrainPredictionResponse)
def predict(payload: GrainPredictionRequest, request: Request) -> GrainPredictionResponse:
    """Predict daily grain_kg and emit one structured JSON log line to stdout."""
    started = time.perf_counter()
    model = getattr(request.app.state, "model", None)
    if model is None:
        model = joblib.load(MODEL_PATH)
        request.app.state.model = model

    features = pd.DataFrame([payload.model_dump()])
    prediction = float(model.predict(features)[0])
    latency_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "prediction_served",
        extra={
            "prediction": round(prediction, 4),
            "latency_ms": round(latency_ms, 3),
            "status_code": 200,
        },
    )
    return GrainPredictionResponse(prediction=prediction)
