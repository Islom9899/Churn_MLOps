"""Model loading helpers for the serving API.

MLOps step: Phase 3, production serving.

This module keeps FastAPI independent from the exact model source. In local
development it can fall back to a joblib artifact, while production can force
MLflow Model Registry loading by setting ALLOW_LOCAL_MODEL_FALLBACK=false.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.validation import rename_csv_columns

DEFAULT_LOCAL_MODEL_PATH = Path("models/churn_model.joblib")
DEFAULT_MODEL_NAME = "churn-prediction"
DEFAULT_MODEL_ALIAS = "production"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment flag with a predictable default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_local_model(path: Path | None = None) -> Any:
    """Load a local joblib model artifact."""
    model_path = path or Path(os.getenv("LOCAL_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH))
    return joblib.load(model_path)


def load_registry_model(
    model_name: str | None = None,
    alias: str | None = None,
    tracking_uri: str | None = None,
) -> Any:
    """Load a sklearn model from MLflow Model Registry by alias."""
    import mlflow
    import mlflow.sklearn

    resolved_name = model_name or os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    resolved_alias = alias or os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
    resolved_tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
    )

    mlflow.set_tracking_uri(resolved_tracking_uri)
    model_uri = f"models:/{resolved_name}@{resolved_alias}"
    return mlflow.sklearn.load_model(model_uri)


def load_serving_model() -> Any:
    """Load the model configured for the API process."""
    source = os.getenv("MODEL_SOURCE", "registry").strip().lower()
    allow_fallback = _env_bool("ALLOW_LOCAL_MODEL_FALLBACK", default=True)

    if source == "local":
        return load_local_model()

    try:
        return load_registry_model()
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Could not load the production model from MLflow Model Registry."
            ) from exc
        return load_local_model()


def expected_feature_columns(model: Any) -> list[str]:
    """Return feature names expected by the fitted sklearn pipeline."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    named_steps = getattr(model, "named_steps", {})
    preprocessor = named_steps.get("preprocess") if named_steps else None
    transformers = getattr(preprocessor, "transformers", [])

    columns: list[str] = []
    for _, _, transformer_columns in transformers:
        if isinstance(transformer_columns, (list, tuple)):
            columns.extend(str(column) for column in transformer_columns)
    return columns


def build_prediction_frame(model: Any, payload: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame with columns matching the loaded model."""
    raw_frame = pd.DataFrame([payload])
    snake_frame = rename_csv_columns(raw_frame)
    expected_columns = expected_feature_columns(model)

    if not expected_columns:
        return snake_frame

    if set(expected_columns).issubset(raw_frame.columns):
        return raw_frame[expected_columns]

    if set(expected_columns).issubset(snake_frame.columns):
        return snake_frame[expected_columns]

    missing = sorted(set(expected_columns) - set(raw_frame.columns) - set(snake_frame.columns))
    raise ValueError(f"Model input is missing required feature columns: {missing}")
