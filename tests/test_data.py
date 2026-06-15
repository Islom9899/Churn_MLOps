"""
test_data.py — src/train.py dagi trening funksiyalarini tekshiradi.

Phase 2 yangiligi:
  - snake_case ustun nomlari (training_dataset.csv formatiga mos)
  - clean_data yo'q — ma'lumot Phase 1 da tozalangan
  - build_pipeline() argumentsiz chaqiriladi (ustun ro'yxatlari modul ichida)
"""

import pandas as pd
from src.train import (
    TARGET,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    META_COLUMNS,
    split_features_target,
    build_pipeline,
)


# ---- Minimal to'liq test DataFrame yaratuvchi yordamchi ----
def _make_df(n: int = 6) -> pd.DataFrame:
    """Barcha 19 belgi + meta + yorliq ustunlari bilan kichik DataFrame."""
    return pd.DataFrame({
        # Meta ustunlar (trening uchun kerak emas)
        "customer_id": [f"C{i}" for i in range(n)],
        "feature_timestamp": ["2023-06-01"] * n,
        "label_timestamp": ["2023-07-01"] * n,

        # Kategorik belgilar
        "gender": ["Male", "Female"] * (n // 2),
        "partner": ["Yes", "No"] * (n // 2),
        "dependents": ["No", "Yes"] * (n // 2),
        "phone_service": ["Yes"] * n,
        "multiple_lines": ["No"] * n,
        "internet_service": ["DSL", "Fiber optic"] * (n // 2),
        "online_security": ["No"] * n,
        "online_backup": ["Yes"] * n,
        "device_protection": ["No"] * n,
        "tech_support": ["No"] * n,
        "streaming_tv": ["No"] * n,
        "streaming_movies": ["No"] * n,
        "contract": ["Month-to-month", "One year"] * (n // 2),
        "paperless_billing": ["Yes"] * n,
        "payment_method": ["Electronic check"] * n,

        # Raqamli belgilar
        "senior_citizen": [0, 0, 1, 0, 0, 1][:n],
        "tenure": [1, 50, 5, 60, 2, 40][:n],
        "monthly_charges": [20.0, 90.0, 30.0, 100.0, 25.0, 80.0][:n],
        "total_charges": [20.0, 4500.0, 150.0, 6000.0, 50.0, 3200.0][:n],

        # Yorliq (muvozanatli: 3 ta 0, 3 ta 1)
        "churn": [0, 1, 0, 1, 0, 1][:n],
    })


# ---- Testlar ----

def test_split_drops_meta_and_target():
    """split_features_target: meta ustunlar va yorliq X dan olib tashlanadi."""
    df = _make_df()
    X, y = split_features_target(df)

    # Yorliq X da bo'lmasligi kerak
    assert TARGET not in X.columns

    # Meta ustunlar X da bo'lmasligi kerak
    for col in META_COLUMNS:
        assert col not in X.columns, f"Meta ustun X da qoldi: {col}"

    # Barcha 19 belgi X da bo'lishi kerak
    assert set(FEATURE_COLUMNS).issubset(set(X.columns))

    # y to'g'ri yorliqlarni o'z ichiga oladi
    assert list(y) == [0, 1, 0, 1, 0, 1]


def test_split_y_is_integer_series():
    """split_features_target: y ustuni integer tipida bo'lishi kerak."""
    df = _make_df()
    _, y = split_features_target(df)
    assert y.dtype in ("int64", "int32", "int8"), f"Kutilmagan tur: {y.dtype}"


def test_build_pipeline_fits_and_predicts():
    """build_pipeline: to'g'ri fit/predict qiladi, natija 0 yoki 1."""
    df = _make_df()
    X, y = split_features_target(df)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y), "Bashorat soni kirish soni bilan bir xil bo'lishi kerak"
    assert set(preds.tolist()) <= {0, 1}, "Bashorat faqat 0 yoki 1 bo'lishi kerak"


def test_build_pipeline_returns_probabilities():
    """predict_proba 0..1 oralig'ida bo'lishi kerak."""
    df = _make_df()
    X, y = split_features_target(df)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    probas = pipeline.predict_proba(X)[:, 1]   # "ketadi" ehtimoli
    assert (probas >= 0.0).all(), "Ehtimol manfiy bo'lishi mumkin emas"
    assert (probas <= 1.0).all(), "Ehtimol 1 dan oshishi mumkin emas"


def test_categorical_and_numeric_no_overlap():
    """CATEGORICAL_FEATURES va NUMERIC_FEATURES kesishmaydi."""
    overlap = set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)
    assert not overlap, f"Bir xil ustun ikkala ro'yxatda: {overlap}"


def test_feature_columns_complete():
    """FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES."""
    assert set(FEATURE_COLUMNS) == set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
    assert len(FEATURE_COLUMNS) == 19, f"19 belgi kutildi, {len(FEATURE_COLUMNS)} topildi"
