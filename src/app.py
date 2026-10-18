"""FastAPI service: /predict, /health, /metrics with request logging."""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from preprocess import FEATURES

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
STATE = {"model": None}

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency",
                    ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["label"])


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=30, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: Optional[int] = Field(None, ge=0, le=3)
    thal: Optional[int] = Field(None, ge=3, le=7)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if Path(MODEL_PATH).exists():
        STATE["model"] = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    else:
        log.error("Model file not found at %s", MODEL_PATH)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    route = request.scope.get("route")
    endpoint = getattr(route, "path", "unmatched")
    REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
    LATENCY.labels(endpoint).observe(elapsed)
    log.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
             response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.post("/predict")
def predict(patient: Patient):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
    p_disease = float(model.predict_proba(X)[0, 1])
    pred = int(p_disease >= 0.5)
    confidence = p_disease if pred == 1 else 1 - p_disease
    label = "heart_disease" if pred == 1 else "no_heart_disease"
    PREDICTIONS.labels(label).inc()
    log.info("prediction=%s confidence=%.3f input=%s", pred, confidence,
             patient.model_dump())
    return {"prediction": pred, "label": label,
            "confidence": round(confidence, 4),
            "probability_disease": round(p_disease, 4)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
