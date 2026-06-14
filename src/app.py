"""
app.py — Churn modelini FastAPI orqali HTTP xizmat (serving) qilish.

Vazifasi:
    Saqlangan modelni (models/churn_model.joblib) yuklaydi va 2 ta endpoint beradi:
      GET  /health   -> xizmat tirikmi (sog'liq tekshiruvi)
      POST /predict  -> bitta mijoz ma'lumotini olib, churn (ketish) bashoratini qaytaradi

Ishga tushirish:   uvicorn src.app:app --reload   (loyiha root'idan!)
Hujjat (Swagger):  http://127.0.0.1:8000/docs      (shu yerdan to'g'ridan-to'g'ri sinaysan)
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---- Modelni BIR MARTA yuklaymiz ---------------------------------------------
# Server ko'tarilganda 1 marta yuklanadi (har so'rovda qayta emas) — tez va samarali.
# Diqqat: avval `python src/train.py` ishlatib model yaratilgan bo'lishi shart.
MODEL_PATH = Path("models/churn_model.joblib")
model = joblib.load(MODEL_PATH)


# ---- Kirish sxemasi: bitta mijozning belgilari -------------------------------
# Pydantic avtomat tekshiradi: maydon yetishmasa yoki turi noto'g'ri bo'lsa -> 422 xato.
# Maydon NOMLARI dataset ustun nomlari bilan AYNAN bir xil (model shu nomlarni kutadi).
# Raqamli: SeniorCitizen, tenure (int); MonthlyCharges, TotalCharges (float). Qolgani matn (str).
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

    # /docs sahifasidagi "Try it out" tugmasi shu namunani avtomat to'ldiradi:
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


# ---- Chiqish sxemasi: bashorat javobi ----------------------------------------
class PredictionResponse(BaseModel):
    churn: int = Field(description="0 = qoladi, 1 = ketadi")
    churn_probability: float = Field(description="Ketish ehtimoli (0..1)")
    churn_label: str = Field(description="'Yes' yoki 'No'")


# ---- FastAPI ilovasi ---------------------------------------------------------
app = FastAPI(title="Churn Prediction API", version="1.0")

@app.get("/")
def root() -> dict:
    """Asosiy sahifa — foydalanuvchini kutib oladi va /docs ga yo'naltiradi."""
    return {"message": "Welcome to the Churn Prediction API! Visit /docs for usage."}
@app.get("/health")
def health() -> dict:
    """Sog'liq tekshiruvi — xizmat ishlayotganini bildiradi (Docker/monitoring shuni so'raydi)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Bitta mijoz ma'lumotini olib, churn bashoratini qaytaradi."""
    # Pydantic obyektini 1 qatorli DataFrame ga aylantiramiz (model DataFrame kutadi).
    X = pd.DataFrame([features.model_dump()])

    # Pipeline ichida one-hot + RandomForest bor — tozalash/kodlash avtomat qo'llanadi.
    proba = float(model.predict_proba(X)[0, 1])  # "ketadi" ehtimoli
    pred = int(model.predict(X)[0])              # 0 yoki 1

    return PredictionResponse(
        churn=pred,
        churn_probability=round(proba, 4),
        churn_label="Yes" if pred == 1 else "No",
    )