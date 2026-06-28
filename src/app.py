"""FastAPI serving layer for churn prediction.

MLOps step: Phase 3, production serving.

Endpoints:
    GET  /        -> short service metadata
    GET  /health  -> liveness check
    POST /predict -> churn prediction from customer features
    POST /ingest  -> labeled customer record for future retraining
"""

from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.validation import rename_csv_columns
from src.db import SessionLocal
from src.model_loader import build_prediction_frame, load_serving_model
from src.models import ChurnLabel, CustomerFeature


class CustomerFeatures(BaseModel):
    """Prediction request schema accepted by the public API."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0, le=500.0)
    TotalCharges: float = Field(ge=0.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class CustomerRecord(CustomerFeatures):
    """Labeled observation persisted for future retraining."""

    Churn: Literal["Yes", "No", "1", "0"]
    customer_id: str | None = Field(default=None, min_length=1)
    feature_timestamp: datetime | None = None
    label_timestamp: datetime | None = None


class PredictionResponse(BaseModel):
    """Prediction response returned by /predict."""

    churn: int = Field(description="0 = qoladi, 1 = ketadi")
    churn_probability: float = Field(description="Ketish ehtimoli (0..1)")
    churn_label: str = Field(description="'Yes' yoki 'No'")


app = FastAPI(title="Churn Prediction API", version="3.0")
model = load_serving_model()


def _churn_to_int(value: str) -> int:
    """Normalize API churn labels to the integer training format."""
    normalized = value.strip().lower()
    if normalized in {"yes", "1"}:
        return 1
    if normalized in {"no", "0"}:
        return 0
    raise HTTPException(status_code=422, detail="Churn must be Yes/No or 1/0.")


def _feature_payload(record: CustomerFeatures) -> dict:
    """Convert API field names to the snake_case DB feature format."""
    raw_frame = pd.DataFrame([record.model_dump()])
    snake_row = rename_csv_columns(raw_frame).iloc[0]
    return snake_row.to_dict()


def _timestamps(record: CustomerRecord) -> tuple[datetime, datetime]:
    """Return PIT-safe feature and label timestamps for ingestion."""
    label_ts = record.label_timestamp or datetime.now(timezone.utc)
    feature_ts = record.feature_timestamp or label_ts - timedelta(days=30)

    if feature_ts >= label_ts:
        raise HTTPException(
            status_code=422,
            detail="feature_timestamp must be earlier than label_timestamp.",
        )
    return feature_ts, label_ts


@app.get("/")
def root() -> dict:
    """Return small API metadata for load balancers and humans."""
    return {"service": "churn-prediction-api", "status": "ok"}


@app.get("/health")
def health() -> dict:
    """Return service liveness status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Return churn prediction for one customer."""
    X = build_prediction_frame(model, features.model_dump())
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return PredictionResponse(
        churn=pred,
        churn_probability=round(proba, 4),
        churn_label="Yes" if pred == 1 else "No",
    )


@app.post("/ingest")
def ingest(record: CustomerRecord) -> dict:
    """Persist one labeled observation into production PIT tables."""
    customer_id = record.customer_id or f"api-{uuid4().hex}"
    feature_ts, label_ts = _timestamps(record)
    features = _feature_payload(record)

    customer_feature = CustomerFeature(
        customer_id=customer_id,
        feature_timestamp=feature_ts,
        gender=features["gender"],
        senior_citizen=int(features["senior_citizen"]),
        partner=features["partner"],
        dependents=features["dependents"],
        tenure=int(features["tenure"]),
        phone_service=features["phone_service"],
        multiple_lines=features["multiple_lines"],
        internet_service=features["internet_service"],
        online_security=features["online_security"],
        online_backup=features["online_backup"],
        device_protection=features["device_protection"],
        tech_support=features["tech_support"],
        streaming_tv=features["streaming_tv"],
        streaming_movies=features["streaming_movies"],
        contract=features["contract"],
        paperless_billing=features["paperless_billing"],
        payment_method=features["payment_method"],
        monthly_charges=float(features["monthly_charges"]),
        total_charges=float(features["total_charges"]),
    )
    churn_label = ChurnLabel(
        customer_id=customer_id,
        churn=_churn_to_int(record.Churn),
        label_timestamp=label_ts,
    )

    with SessionLocal() as session:
        session.add(customer_feature)
        session.add(churn_label)
        session.commit()
        session.refresh(customer_feature)
        session.refresh(churn_label)
        feature_id = customer_feature.id
        label_id = churn_label.id

    return {
        "status": "saved",
        "customer_id": customer_id,
        "feature_id": feature_id,
        "label_id": label_id,
    }
