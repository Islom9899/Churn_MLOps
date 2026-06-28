"""Tests for serving model input adaptation."""

from src.model_loader import build_prediction_frame


class FakeModel:
    def __init__(self, columns):
        self.feature_names_in_ = columns


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


def test_build_prediction_frame_keeps_pascal_case_for_legacy_model():
    """Local legacy artifacts still receive the columns they were trained with."""
    model = FakeModel(["gender", "SeniorCitizen", "Partner", "MonthlyCharges"])
    frame = build_prediction_frame(model, VALID)

    assert frame.columns.tolist() == ["gender", "SeniorCitizen", "Partner", "MonthlyCharges"]


def test_build_prediction_frame_converts_to_snake_case_for_registry_model():
    """Registry models trained by the current pipeline receive snake_case columns."""
    model = FakeModel(["gender", "senior_citizen", "partner", "monthly_charges"])
    frame = build_prediction_frame(model, VALID)

    assert frame.columns.tolist() == ["gender", "senior_citizen", "partner", "monthly_charges"]
    assert frame.iloc[0]["senior_citizen"] == 0
