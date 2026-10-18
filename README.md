# Heart Disease Risk Prediction - MLOps Assignment 1

End-to-end pipeline: data -> EDA -> model training (MLflow) -> tests/CI -> Docker -> Kubernetes -> monitoring.

## Layout
```
src/            download_data.py, preprocess.py, eda.py, train.py, predict.py, app.py
notebooks/      01_eda.ipynb
test/           pytest unit tests (preprocessing, model, API)
k8s/            deployment.yaml, service.yaml, monitoring.yaml (Prometheus + Grafana)
.github/workflows/ci.yml   lint -> test -> train -> docker build + smoke test
screenshots/    evidence for the report
docs/           report template
```

## 1. Setup (Python 3.11)
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data, EDA, training
```bash
python src/download_data.py     # downloads UCI Cleveland data -> data/heart.csv
python src/eda.py               # plots -> screenshots/eda/   (or open notebooks/01_eda.ipynb)
python src/train.py             # trains LR + RF, logs to MLflow, saves models/model.joblib
mlflow ui                       # http://localhost:5000
python src/predict.py           # sample inference
```

## 3. Tests and lint
```bash
flake8 src test
pytest -v
```

## 4. Run the API locally
```bash
PYTHONPATH=src uvicorn app:app --port 8000      # Windows PowerShell: $env:PYTHONPATH="src"
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Endpoints: `POST /predict`, `GET /health`, `GET /metrics`, docs at `/docs`.

## 5. Docker
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
```

## 6. Kubernetes (Docker Desktop or Minikube)
```bash
# Minikube only: make the local image available to the cluster
minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
# Minikube: run `minikube tunnel` in another terminal, or `minikube service heart-api --url`
# Docker Desktop: service is on http://localhost:80
```

## 7. Monitoring
```bash
kubectl apply -f k8s/monitoring.yaml
# Prometheus: :9090 | Grafana: :3000 (admin/admin) -> add data source http://prometheus:9090
# Useful queries: rate(api_requests_total[1m]), predictions_total,
#   histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))
kubectl logs -l app=heart-api      # request logs
```
