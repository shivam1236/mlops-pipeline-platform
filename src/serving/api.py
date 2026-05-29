"""
Model Serving API
FastAPI-based inference endpoint with health checks and metrics.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import time
import os
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(
    title="MLOps Model Serving",
    description="Production model inference API",
    version="1.0.0",
)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "model_predictions_total", "Total predictions", ["model_version", "status"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds", "Prediction latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
DRIFT_SCORE = Histogram(
    "model_input_drift_score", "Input drift score per request"
)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")

model = None


def load_model():
    """Load model from disk."""
    global model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded: {MODEL_VERSION} from {MODEL_PATH}")


class PredictionRequest(BaseModel):
    """Input features for prediction."""
    features: list[list[float]]
    request_id: str | None = None


class PredictionResponse(BaseModel):
    """Prediction output."""
    predictions: list[int]
    probabilities: list[list[float]]
    model_version: str
    latency_ms: float
    request_id: str | None = None


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run inference on input features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features = np.array(request.features)

        # Predict
        predictions = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()

        latency = (time.time() - start_time) * 1000

        # Record metrics
        PREDICTION_COUNT.labels(model_version=MODEL_VERSION, status="success").inc(len(predictions))
        PREDICTION_LATENCY.observe(latency / 1000)

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_version=MODEL_VERSION,
            latency_ms=round(latency, 2),
            request_id=request.request_id,
        )

    except Exception as e:
        PREDICTION_COUNT.labels(model_version=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/reload")
async def reload_model():
    """Hot-reload model without downtime."""
    try:
        load_model()
        return {"status": "reloaded", "model_version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
