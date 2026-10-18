# MLOps Assignment 1 - Report (target ~10 pages)

Name / Roll no: ____   |   Repository: <GitHub URL>   |   API access: <URL or local instructions>

## 1. Setup and install instructions
(Python version, venv, `pip install -r requirements.txt`, commands to run each stage - see README.)

## 2. Data and EDA
- Dataset source, size, features, target definition (0 vs 1-4 -> binary).
- Missing values found (ca, thal) and how you handled them.
- Figures: class balance, histograms, correlation heatmap (from screenshots/eda/).
- 3-5 observations YOU draw from the plots.

## 3. Feature engineering and modelling choices
- Preprocessing pipeline (imputation, scaling, one-hot) and why.
- Models, hyperparameter grids, CV scheme (5-fold stratified), selection criterion (ROC-AUC).
- Results table: accuracy / precision / recall / ROC-AUC for each model (paste from models/metrics.json).
- Why you chose the final model; discuss recall vs precision in a medical setting.

## 4. Experiment tracking (MLflow)
- What is logged (params, metrics, ROC/confusion plots, model). Screenshot of MLflow UI run comparison.

## 5. Architecture diagram
(Draw: data -> train -> MLflow -> model.joblib -> Docker image -> Kubernetes (Deployment/Service) -> Prometheus -> Grafana; GitHub Actions on the left.)

## 6. CI/CD
- Workflow stages (lint, test, train, docker). Screenshots: successful run, one deliberately failing run, uploaded artifacts.

## 7. Containerization and deployment
- Docker build/run proof, curl /predict output, `kubectl get pods,svc`, service reachable.

## 8. Monitoring and logging
- Request logs, /metrics output, Prometheus targets page, Grafana dashboard screenshots.

## 9. Conclusion, limitations, future work

## Screenshot checklist (save in screenshots/)
- [ ] EDA plots  - [ ] MLflow UI  - [ ] pytest passing  - [ ] flake8 clean
- [ ] GitHub Actions green run + failed run + artifacts
- [ ] docker build + docker run + curl /predict
- [ ] kubectl get pods,svc + curl against the service
- [ ] Prometheus targets + Grafana dashboard + API logs
