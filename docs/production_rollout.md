# Production rollout guide

This document explains the production path added in Phase 3. It is written as a
step-by-step operator guide plus code notes for the changed files.

## MLOps release steps

1. `python -m pip install -r requirements.txt`
   - Installs training, testing, DB, DVC, Prefect, and MLflow dependencies.

2. `python -m src.data.prepare`
   - Creates the PostgreSQL tables.
   - Seeds `customer_features` and `churn_labels`.
   - Exports `data/training_dataset.csv`.
   - Validates the training dataset before it can be used.

3. `python -m src.pipeline.train_flow`
   - Loads the DVC-tracked training dataset.
   - Validates it with Pandera.
   - Trains the sklearn pipeline.
   - Logs parameters and metrics to MLflow.
   - Registers the model as `churn-prediction`.
   - Assigns the newest passing model to the `staging` alias.

4. `python -m src.pipeline.promote_model`
   - Reads the model currently behind the `staging` alias.
   - Loads the MLflow run metrics for that model version.
   - Checks `roc_auc >= 0.78` and `f1 >= 0.50`.
   - Assigns the same model version to the `production` alias if it passes.

5. `uvicorn src.app:app --host 0.0.0.0 --port 8000`
   - Starts the FastAPI service.
   - Loads `models:/churn-prediction@production` when registry mode is enabled.
   - Keeps local `models/churn_model.joblib` fallback only for development.

6. `docker build -t churn-api .`
   - Builds a production API image.
   - Installs only serving dependencies from `requirements-serve.txt`.
   - Copies the API source code.
   - Requires a reachable MLflow registry at runtime.

7. `docker run --env DATABASE_URL=... --env MLFLOW_TRACKING_URI=... -p 8000:8000 churn-api`
   - Starts the production container.
   - Uses `MODEL_SOURCE=registry`.
   - Uses `MLFLOW_MODEL_ALIAS=production`.
   - Disables local model fallback.

## File roles by MLOps phase

| File | MLOps step | What it does |
| --- | --- | --- |
| `src/data/seed.py` | Phase 1 data layer | Loads raw Telco CSV and writes feature and label tables. |
| `src/data/pit_join.py` | Phase 1 data layer | Builds point-in-time training data without label leakage. |
| `src/data/validation.py` | Phase 1 data quality | Defines schema checks for feature and training data. |
| `src/train.py` | Phase 2 training | Builds, trains, and evaluates the sklearn model pipeline. |
| `src/pipeline/train_flow.py` | Phase 2 orchestration | Runs the training workflow and registers the model to MLflow. |
| `src/pipeline/promote_model.py` | Phase 3 release gate | Promotes `staging` to `production` only when metrics pass. |
| `src/model_loader.py` | Phase 3 serving | Loads the model from MLflow Registry or local fallback. |
| `src/app.py` | Phase 3 serving | Serves `/predict`, `/ingest`, `/health`, and `/`. |
| `src/observability.py` | Phase 4 observability | Exposes Prometheus metrics and optional OpenTelemetry tracing. |
| `src/db.py` | Shared infra | Creates the SQLAlchemy engine and session factory. |
| `src/models.py` | Shared infra | Defines DB tables used by training and serving. |
| `.github/workflows/ci.yml` | CI/CD | Trains, promotes, and tests on every push or pull request. |
| `Dockerfile` | Deployment | Builds the production API container. |
| `requirements-serve.txt` | Deployment | Pins runtime dependencies for the API image. |
| `.dvcignore` | Data ops | Prevents DVC from scanning local caches and runtime artifacts. |
| `k8s/base/` | Phase 5 platform | Deploys the API to Kubernetes with probes and autoscaling. |
| `monitoring/grafana/` | Phase 4 observability | Provides a Grafana dashboard template. |
| `monitoring/prometheus/` | Phase 4 observability | Provides a Prometheus Operator ServiceMonitor. |

## Code notes: `src/model_loader.py`

- Module docstring: states this file belongs to Phase 3 serving.
- `from __future__ import annotations`: allows modern type hints safely.
- `import os`: reads deployment configuration from environment variables.
- `from pathlib import Path`: handles local model paths portably.
- `from typing import Any`: allows sklearn and MLflow model objects without tight typing.
- `import joblib`: loads legacy local `.joblib` artifacts for development.
- `import pandas as pd`: builds one-row prediction DataFrames.
- `rename_csv_columns`: converts public API PascalCase names to training snake_case names.
- `DEFAULT_*` constants: define the local artifact, registry model name, alias, and tracking URI.
- `_env_bool`: normalizes environment strings such as `true`, `1`, and `yes`.
- `load_local_model`: loads `models/churn_model.joblib` or `LOCAL_MODEL_PATH`.
- `load_registry_model`: loads `models:/<name>@<alias>` from MLflow Model Registry.
- `load_serving_model`: selects registry or local mode and applies local fallback policy.
- `expected_feature_columns`: reads feature names from a fitted sklearn pipeline.
- `build_prediction_frame`: creates the exact DataFrame columns required by the loaded model.

## Code notes: `src/pipeline/promote_model.py`

- Module docstring: states this file is the production release gate.
- MLflow imports: create a client connected to the configured tracking URI.
- Training constants import: reuses the same model name and metric thresholds as training.
- `PRODUCTION_ALIAS`: defines the alias the API uses in production.
- `metrics_pass`: returns `True` only when both required metrics pass.
- `promote_model`: reads the `staging` version, checks metrics, and sets the `production` alias.
- `if __name__ == "__main__"`: lets the file run as `python -m src.pipeline.promote_model`.

## Code notes: `src/app.py`

- Module docstring: states this file is the Phase 3 serving layer.
- `datetime`, `timedelta`, `timezone`: create PIT-safe ingestion timestamps.
- `Literal`: restricts API strings to valid category values.
- `uuid4`: creates a customer id when the caller does not send one.
- `pandas`: converts request bodies into DataFrame rows for DB and model input.
- `FastAPI`, `HTTPException`: define the API and return validation errors.
- `BaseModel`, `Field`: define request and response schemas.
- `rename_csv_columns`: converts public request names to DB snake_case names.
- `SessionLocal`: opens database sessions for ingestion.
- `build_prediction_frame`, `load_serving_model`: isolate model source and feature format logic.
- `ChurnLabel`, `CustomerFeature`: write production PIT tables instead of the legacy table.
- `CustomerFeatures`: validates the public prediction payload.
- `CustomerRecord`: extends prediction input with `Churn`, `customer_id`, and timestamps.
- `PredictionResponse`: fixes the response contract for `/predict`.
- `app = FastAPI(...)`: creates the API application.
- `model = load_serving_model()`: loads the configured model once at process startup.
- `_churn_to_int`: maps `Yes/No` or `1/0` to the training label format.
- `_feature_payload`: converts API field names into DB/training field names.
- `_timestamps`: ensures `feature_timestamp < label_timestamp` for PIT correctness.
- `root`: gives `/` a real response for humans and load balancers.
- `health`: returns a liveness response.
- `predict`: builds model input, calls `predict_proba`, calls `predict`, and returns labels.
- `ingest`: writes one feature snapshot and one label row for future retraining.

## Code notes: deployment files

- `Dockerfile` line `FROM python:3.13-slim`: uses a small Python runtime base.
- `WORKDIR /app`: sets the container working directory.
- `COPY requirements-serve.txt .`: copies runtime dependency pins first for Docker cache reuse.
- `RUN pip install --no-cache-dir -r requirements-serve.txt`: installs API dependencies.
- `COPY src/ ./src/`: copies source code into the image.
- `ENV MODEL_SOURCE=registry`: forces registry loading in the image.
- `ENV MLFLOW_MODEL_ALIAS=production`: makes the API use the production model.
- `ENV ALLOW_LOCAL_MODEL_FALLBACK=false`: prevents accidental local fallback in production.
- `EXPOSE 8000`: documents the service port.
- `CMD [...]`: starts Uvicorn without development reload.

- `requirements-serve.txt`: includes FastAPI, Uvicorn, sklearn/joblib, MLflow, Pandera,
  SQLAlchemy, psycopg2, and python-dotenv because the serving process now loads registry
  models, imports shared validation helpers, and writes DB rows.

- `.github/workflows/ci.yml`: now runs training, then promotion, then tests.

- `.dvcignore`: ignores local runtime caches so DVC status does not scan non-data folders.

## Code notes: `src/observability.py`

- Module docstring: states this file belongs to Phase 4 observability.
- `prometheus_client` imports: define counters and histograms for monitoring.
- `HTTP_REQUESTS`: counts HTTP traffic by method, path, and status.
- `HTTP_REQUEST_SECONDS`: records request latency as a histogram.
- `PREDICTIONS`: counts predictions by churn label.
- `PREDICTION_PROBABILITY`: records the probability distribution returned by the model.
- `INGESTED_RECORDS`: counts accepted labeled records.
- `setup_prometheus`: adds request middleware and exposes `/metrics`.
- `record_prediction`: records one model prediction after `/predict`.
- `record_ingest`: records one accepted observation after `/ingest`.
- `setup_opentelemetry`: enables FastAPI and SQLAlchemy tracing only when `ENABLE_OTEL=true`.

## Code notes: Kubernetes and monitoring files

- `k8s/base/namespace.yaml`: creates the `churn-mlops` namespace.
- `k8s/base/deployment.yaml`: runs two `churn-api` replicas with probes, resource limits, registry model mode, and OTLP env vars.
- `k8s/base/service.yaml`: exposes the API inside the cluster.
- `k8s/base/hpa.yaml`: scales the API by CPU utilization.
- `k8s/base/kustomization.yaml`: lets `kubectl apply -k k8s/base` deploy the base stack.
- `monitoring/prometheus/churn-api-servicemonitor.yaml`: tells Prometheus Operator to scrape `/metrics`.
- `monitoring/grafana/churn-api-dashboard.json`: visualizes request rate, errors, latency, predictions, ingest rate, and probability distribution.
