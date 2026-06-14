"""
test_app.py — FastAPI endpointlarni tekshiradi (TestClient bilan, haqiqiy server ko'tarmasdan).
Eslatma: bu testlar uchun model mavjud bo'lishi kerak (avval `python src/train.py`).
"""

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

VALID = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def test_health_ok():
    """/health 200 va status ok qaytaradi."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_returns_prediction():
    """/predict to'g'ri kirishda 200 va to'g'ri formatdagi javob beradi."""
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_label"] in ("Yes", "No")
    # churn va churn_label mos kelishi kerak
    assert (body["churn"] == 1) == (body["churn_label"] == "Yes")


def test_predict_missing_field_returns_422():
    """Maydon yetishmasa Pydantic 422 qaytaradi (himoya ishlaydi)."""
    bad = dict(VALID)
    del bad["tenure"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422