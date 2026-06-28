"""FastAPI endpoint tests."""

from fastapi.testclient import TestClient

from src.app import app
from src.models import ChurnLabel, CustomerFeature

client = TestClient(app)

VALID = {
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


def test_health_ok():
    """/health returns liveness status."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_ok():
    """/ returns service metadata."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "churn-prediction-api"


def test_predict_valid_returns_prediction():
    """/predict returns the public response contract."""
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_label"] in ("Yes", "No")
    assert (body["churn"] == 1) == (body["churn_label"] == "Yes")


def test_predict_missing_field_returns_422():
    """Missing feature fields are rejected by Pydantic."""
    bad = dict(VALID)
    del bad["tenure"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_ingest_writes_feature_and_label(monkeypatch):
    """/ingest writes production PIT table objects, not the legacy table."""
    saved = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add(self, obj):
            obj.id = len(saved) + 1
            saved.append(obj)

        def commit(self):
            return None

        def refresh(self, obj):
            return None

    monkeypatch.setattr("src.app.SessionLocal", lambda: FakeSession())

    payload = {
        **VALID,
        "customer_id": "api-test-customer",
        "Churn": "Yes",
        "feature_timestamp": "2024-01-01T00:00:00+00:00",
        "label_timestamp": "2024-02-01T00:00:00+00:00",
    }
    r = client.post("/ingest", json=payload)

    assert r.status_code == 200
    assert r.json()["customer_id"] == "api-test-customer"
    assert isinstance(saved[0], CustomerFeature)
    assert isinstance(saved[1], ChurnLabel)
    assert saved[0].customer_id == "api-test-customer"
    assert saved[1].churn == 1


def test_ingest_rejects_non_pit_timestamp_order():
    """feature_timestamp must be earlier than label_timestamp."""
    payload = {
        **VALID,
        "Churn": "No",
        "feature_timestamp": "2024-02-01T00:00:00+00:00",
        "label_timestamp": "2024-01-01T00:00:00+00:00",
    }
    r = client.post("/ingest", json=payload)
    assert r.status_code == 422
