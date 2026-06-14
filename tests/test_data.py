"""
test_data.py — train.py dagi ma'lumot tayyorlash mantig'ini tekshiradi.
Bu yerda fayl (CSV) kerak emas — kichik sun'iy jadval bilan mantiqni sinaymiz (tez, ishonchli).
"""

import pandas as pd

from src.train import clean_data, build_pipeline, TARGET


def test_clean_data_drops_id_fixes_charges_and_maps_target():
    """clean_data: customerID o'chadi, TotalCharges raqam (bo'sh->0), Churn -> 0/1."""
    raw = pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3"],
            "TotalCharges": ["29.85", " ", "100.5"],  # o'rtadagi bo'sh -> 0 bo'lishi kerak
            "MonthlyCharges": [29.85, 50.0, 100.5],
            "Churn": ["Yes", "No", "Yes"],
        }
    )

    cleaned = clean_data(raw)

    # 1) customerID olib tashlangan
    assert "customerID" not in cleaned.columns
    # 2) TotalCharges raqamli va NaN yo'q; bo'sh katak 0 ga aylangan
    assert cleaned["TotalCharges"].dtype != object
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["TotalCharges"].iloc[1] == 0.0
    # 3) Churn 0/1 ga aylangan
    assert cleaned[TARGET].tolist() == [1, 0, 1]

    # 4) clean_data asl jadvalni buzmaydi (copy ishlatadi)
    assert "customerID" in raw.columns


def test_build_pipeline_fits_and_predicts():
    """build_pipeline: Pipeline qaytaradi, fit/predict ishlaydi, natija 0/1."""
    raw = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "tenure": [1, 50, 5, 60, 2, 40],
            "MonthlyCharges": [20.0, 90.0, 30.0, 100.0, 25.0, 80.0],
            "TotalCharges": [20.0, 4500.0, 150.0, 6000.0, 50.0, 3200.0],
            "Churn": [1, 0, 1, 0, 1, 0],
        }
    )

    pipe = build_pipeline(raw)
    X = raw.drop(columns=["Churn"])
    y = raw["Churn"]
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds.tolist()) <= {0, 1}