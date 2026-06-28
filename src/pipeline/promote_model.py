"""Promote a validated MLflow model version to the production alias.

MLOps step: Phase 3, production release gate.

The training flow writes the newest model to the staging alias. This module
checks the model metrics and moves the same version to the production alias
only when it passes the project quality thresholds.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.pipeline.train_flow import (
    MIN_F1,
    MIN_ROC_AUC,
    REGISTERED_MODEL_NAME,
    STAGING_ALIAS,
    TRACKING_URI,
)

PRODUCTION_ALIAS = "production"


def metrics_pass(
    metrics: dict[str, float],
    min_roc_auc: float = MIN_ROC_AUC,
    min_f1: float = MIN_F1,
) -> bool:
    """Return True when metrics are good enough for production serving."""
    return metrics.get("roc_auc", 0.0) >= min_roc_auc and metrics.get("f1", 0.0) >= min_f1


def promote_model(
    model_name: str = REGISTERED_MODEL_NAME,
    source_alias: str = STAGING_ALIAS,
    target_alias: str = PRODUCTION_ALIAS,
    tracking_uri: str = TRACKING_URI,
) -> dict[str, object]:
    """Promote the staging model version to production if metrics pass."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_version = client.get_model_version_by_alias(model_name, source_alias)
    run = client.get_run(model_version.run_id)
    metrics = dict(run.data.metrics)

    if not metrics_pass(metrics):
        raise RuntimeError(
            "Model quality gate failed: "
            f"roc_auc={metrics.get('roc_auc')} min={MIN_ROC_AUC}, "
            f"f1={metrics.get('f1')} min={MIN_F1}"
        )

    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=model_version.version,
    )

    return {
        "model_name": model_name,
        "version": int(model_version.version),
        "source_alias": source_alias,
        "target_alias": target_alias,
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = promote_model()
    print(
        "Promoted "
        f"{result['model_name']} v{result['version']} "
        f"from {result['source_alias']} to {result['target_alias']}"
    )
