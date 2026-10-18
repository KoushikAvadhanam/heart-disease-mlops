
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate the saved model relative to this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run src/train.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational API for a machine-learning classification project.",
    version="1.0.0",
)


# Input fields expected by the trained model.
class PatientData(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(patient: PatientData):
    # Preserve the feature names and order used during training.
    patient_df = pd.DataFrame([patient.model_dump()])

    prediction = int(model.predict(patient_df)[0])
    probabilities = model.predict_proba(patient_df)[0]

    return {
        "prediction": prediction,
        "result": (
            "Disease present" if prediction == 1 else "No disease detected"
        ),
        "probability_disease_present": round(float(probabilities[1]), 4),
        "model": "Logistic Regression",
        "disclaimer": (
            "Educational demonstration only; not a medical diagnosis."
        ),
    }
