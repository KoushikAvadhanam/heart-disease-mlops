
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Heart Disease Prediction API"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict():
    response = client.post("/predict", json=SAMPLE_PATIENT)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["result"] in [
        "Disease present",
        "No disease detected",
    ]
    assert 0.0 <= result["probability_disease_present"] <= 1.0
    assert result["model"] == "Logistic Regression"


def test_predict_invalid_age():
    invalid_patient = SAMPLE_PATIENT.copy()
    invalid_patient["age"] = 150

    response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422
