"""Observability helpers for production serving.

MLOps step: Phase 4, monitoring and tracing.

Prometheus metrics are always available at `/metrics`. OpenTelemetry tracing is
enabled only when `ENABLE_OTEL=true`, so local development and tests do not need
an OTLP collector running.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

HTTP_REQUESTS = Counter(
    "churn_api_http_requests_total",
    "Total HTTP requests handled by the churn API.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_SECONDS = Histogram(
    "churn_api_http_request_duration_seconds",
    "HTTP request latency for the churn API.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "churn_api_predictions_total",
    "Total model predictions by predicted churn label.",
    ["churn_label"],
)
PREDICTION_PROBABILITY = Histogram(
    "churn_api_prediction_probability",
    "Distribution of returned churn probabilities.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
INGESTED_RECORDS = Counter(
    "churn_api_ingested_records_total",
    "Total labeled records accepted by the ingestion endpoint.",
    ["churn"],
)


def setup_prometheus(app: FastAPI) -> None:
    """Attach Prometheus request instrumentation and the `/metrics` endpoint."""

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        path = request.url.path

        if path != "/metrics":
            duration = time.perf_counter() - start
            HTTP_REQUESTS.labels(
                request.method,
                path,
                str(response.status_code),
            ).inc()
            HTTP_REQUEST_SECONDS.labels(request.method, path).observe(duration)

        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_prediction(churn_label: str, probability: float) -> None:
    """Record one model prediction for Prometheus."""
    PREDICTIONS.labels(churn_label).inc()
    PREDICTION_PROBABILITY.observe(probability)


def record_ingest(churn: int) -> None:
    """Record one accepted labeled observation for Prometheus."""
    INGESTED_RECORDS.labels(str(churn)).inc()


def setup_opentelemetry(app: FastAPI, engine: Any | None = None) -> None:
    """Enable OpenTelemetry tracing when the deployment asks for it."""
    if os.getenv("ENABLE_OTEL", "false").strip().lower() not in {"1", "true", "yes"}:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return

    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "churn-api")})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine)
