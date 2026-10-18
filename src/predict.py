"""Load the saved pipeline and predict on one patient record."""
import joblib
import pandas as pd

from preprocess import FEATURES

_model = None


def predict(record: dict):
    global _model
    if _model is None:
        _model = joblib.load("models/model.joblib")
    X = pd.DataFrame([record])[FEATURES].astype(float)
    proba = float(_model.predict_proba(X)[0, 1])
    return {"prediction": int(proba >= 0.5), "confidence": proba if proba >= 0.5 else 1 - proba}


if __name__ == "__main__":
    sample = dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2,
                  thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6)
    print(predict(sample))
