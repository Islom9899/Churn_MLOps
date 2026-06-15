"""
app.py — Churn modelini FastAPI orqali xizmat qilish + ma'lumot yig'ish (ingestion).

Endpointlar:
    GET  /health   -> xizmat tirikmi
    POST /predict  -> mijoz ma'lumotidan churn bashorati (model ishlatadi)
    POST /ingest   -> mijoz yozuvini (belgilar + haqiqiy Churn) bazaga saqlaydi

Ishga tushirish:   uvicorn src.app:app --reload   (loyiha root'idan)
Hujjat (Swagger):  http://127.0.0.1:8000/docs
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.db import SessionLocal     # bazaga yozish uchun sessiya fabrikasi
from src.models import Customer     # customers jadvali (ORM model)

# ---- Modelni BIR MARTA yuklaymiz (server ko'tarilganda) ----
MODEL_PATH = Path("models/churn_model.joblib")
model = joblib.load(MODEL_PATH)


# ---- Kirish sxemasi: mijoz belgilari (bashorat uchun) ----
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 1, "PhoneService": "No",
                "MultipleLines": "No phone service", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85, "TotalCharges": 29.85,
            }
        }
    }


# ---- Ingestion sxemasi: belgilar + haqiqiy Churn natijasi ----
class CustomerRecord(CustomerFeatures):
    """Bazaga saqlash uchun: 19 belgi + haqiqiy Churn ('Yes'/'No')."""
    Churn: str


# ---- Chiqish sxemasi: bashorat javobi ----
class PredictionResponse(BaseModel):
    churn: int = Field(description="0 = qoladi, 1 = ketadi")
    churn_probability: float = Field(description="Ketish ehtimoli (0..1)")
    churn_label: str = Field(description="'Yes' yoki 'No'")


# ---- FastAPI ilovasi ----
app = FastAPI(title="Churn Prediction API", version="2.0")


@app.get("/health")
def health() -> dict:
    """Sog'liq tekshiruvi — xizmat ishlayotganini bildiradi."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Bitta mijoz ma'lumotidan churn bashoratini qaytaradi."""
    X = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return PredictionResponse(
        churn=pred,
        churn_probability=round(proba, 4),
        churn_label="Yes" if pred == 1 else "No",
    )


@app.post("/ingest")
def ingest(record: CustomerRecord) -> dict:
    """Yangi mijoz yozuvini (belgilar + Churn) customers jadvaliga saqlaydi."""
    with SessionLocal() as session:                  # sessiya (avtomat yopiladi)
        customer = Customer(**record.model_dump())   # Pydantic -> ORM obyekt
        session.add(customer)                         # qo'shishga navbatga qo'yamiz
        session.commit()                              # bazaga yozamiz
        session.refresh(customer)                     # id va created_at ni bazadan o'qiymiz
        new_id = customer.id                          # sessiya ichida o'qib olamiz
    return {"status": "saved", "id": new_id}