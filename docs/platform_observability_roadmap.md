# Platform and observability roadmap

This roadmap is the portfolio-grade path for the project after the model can be
trained, registered, promoted, and served.

## Where Grafana, OpenTelemetry, and Kubernetes fit

| Tool | Add in phase | Why it belongs there |
| --- | --- | --- |
| OpenTelemetry | Phase 4: observability instrumentation | The app must emit traces from the first serious production deploy. Add this before Kubernetes rollout so every pod is traceable. |
| Prometheus metrics | Phase 4: observability instrumentation | The API must expose request, latency, prediction, and ingestion metrics before dashboards or alerts can be useful. |
| Grafana | Phase 4: observability dashboards | Grafana is useful after metrics exist. It turns Prometheus data into portfolio-visible production dashboards. |
| Kubernetes | Phase 5: platform deployment | Kubernetes comes after Docker, health checks, metrics, and registry-based model loading are stable. It proves scaling, probes, rolling updates, and secret handling. |
| OpenTelemetry Collector | Phase 5: platform deployment | The collector usually runs inside Kubernetes and receives traces from all pods. |

## Recommended portfolio phases

1. Phase 1: Data foundation
   - PostgreSQL data model.
   - Point-in-time joins.
   - Pandera validation.
   - DVC dataset versioning.

2. Phase 2: Training system
   - Prefect training flow.
   - MLflow experiment tracking.
   - MLflow Model Registry.
   - Dataset hash stored with each run.

3. Phase 3: Production serving
   - FastAPI `/predict`, `/ingest`, `/health`.
   - Production model alias loading from MLflow.
   - Staging-to-production promotion gate.
   - Docker image for serving.

4. Phase 4: Observability
   - Prometheus `/metrics`.
   - Prediction counters and probability distribution.
   - HTTP request rate and latency histograms.
   - Optional OpenTelemetry tracing through `ENABLE_OTEL=true`.
   - Grafana dashboard JSON for request rate, errors, latency, prediction mix, and probability distribution.

5. Phase 5: Kubernetes deployment
   - `Deployment` with two replicas.
   - `Service` for in-cluster traffic.
   - `readinessProbe` and `livenessProbe`.
   - `HorizontalPodAutoscaler`.
   - Secret-based `DATABASE_URL` and `MLFLOW_TRACKING_URI`.
   - Prometheus scrape annotations and `ServiceMonitor`.

6. Phase 6: Cloud CI/CD and artifact storage
   - DVC remote in S3/GCS/Azure/MinIO or another CI-accessible object store.
   - Container registry publishing.
   - GitHub Actions build and deploy.
   - Protected production environment.

7. Phase 7: ML operations maturity
   - Drift detection.
   - Data quality alerting.
   - Model performance monitoring after labels arrive.
   - Canary or shadow model deployment.
   - Rollback to previous MLflow `production` model alias.

## Current implementation in this repo

- `src/observability.py` exposes Prometheus metrics and optional OpenTelemetry tracing.
- `src/app.py` records prediction and ingestion metrics.
- `monitoring/grafana/churn-api-dashboard.json` is the Grafana dashboard template.
- `monitoring/prometheus/churn-api-servicemonitor.yaml` lets Prometheus Operator scrape the API.
- `k8s/base/` contains the Kubernetes base manifests.

## Commands

Run the API locally:

```bash
uvicorn src.app:app --reload
```

Check metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

Deploy Kubernetes manifests after the image and secrets exist:

```bash
kubectl create namespace churn-mlops
kubectl -n churn-mlops create secret generic churn-api-secrets \
  --from-literal=DATABASE_URL="postgresql+psycopg2://..." \
  --from-literal=MLFLOW_TRACKING_URI="https://..."
kubectl apply -k k8s/base
kubectl apply -f monitoring/prometheus/churn-api-servicemonitor.yaml
```

Enable tracing in Kubernetes:

```bash
kubectl -n churn-mlops set env deployment/churn-api ENABLE_OTEL=true
kubectl -n churn-mlops set env deployment/churn-api OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.observability.svc:4318
```

## What a strong 2027 portfolio story should say

This is not just a notebook model. It is an end-to-end production ML system:

- Reproducible data and model lineage with DVC and MLflow.
- Point-in-time training data to prevent leakage.
- Automated training and promotion gates.
- Registry-driven serving with a production alias.
- API observability through Prometheus, Grafana, and OpenTelemetry.
- Kubernetes deployment with probes, autoscaling, and secret-based configuration.
- Clear path to cloud CI/CD, drift monitoring, and canary model rollout.
