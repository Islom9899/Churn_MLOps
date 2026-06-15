"""
train.py — Churn modeli uchun sof trening funksiyalari.

Phase 2 yangiligi:
  - data/training_dataset.csv dan o'qiydi (snake_case, PIT join natijasi)
  - clean_data yo'q — ma'lumot Phase 1 da allaqachon tozalangan va validatsiya qilingan
  - Ustun nomlari aniq ro'yxat (CATEGORICAL_FEATURES, NUMERIC_FEATURES) — select_dtypes emas
  - main() yo'q — orchestratsiya src/pipeline/train_flow.py da
  - Bu modul "sof funksiyalar" — import qilinadi, to'g'ridan-to'g'ri ishga tushirilmaydi

Prefect/MLflow qatlamini src/pipeline/train_flow.py ga qarang.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---- Sozlamalar ----
TRAINING_DATA_PATH = Path("data/training_dataset.csv")   # PIT join natijasi (DVC kuzatadi)
TARGET = "churn"          # yorliq ustuni: integer 0 (qoldi) / 1 (ketdi)
RANDOM_STATE = 42         # takrorlanish uchun — har safar bir xil bo'linish
TEST_SIZE = 0.2           # 20% test uchun
N_ESTIMATORS = 200        # random forest daraxtlar soni

# Meta ustunlar — trening uchun belgi emas, X dan olib tashlanadi
META_COLUMNS = ["customer_id", "feature_timestamp", "label_timestamp"]

# Kategorik belgilar: OneHotEncoder bilan qayta ishlanadi
# Nima uchun aniq ro'yxat? select_dtypes ishonchsiz — raqam sifatida kelgan string bo'lishi mumkin.
# Ro'yxat versiyalanadi va o'zgarsa aniq ko'rinadi.
CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

# Raqamli belgilar: o'zgarmasdan Pipeline ga o'tadi (passthrough)
# senior_citizen 0/1 bo'lsa ham raqamli ko'rib chiqiladi — OneHot shart emas
NUMERIC_FEATURES = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]

# Barcha belgilar birlashtirilgan ro'yxat (ustun tartibini aniq qotirish uchun)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_training_data(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """PIT join natijasi CSV ni o'qiydi.

    Ma'lumot allaqachon tozalangan va validatsiya qilingan (Phase 1).
    Bu funksiya shunchaki diskdan o'qiydi — hech qanday o'zgartirish yo'q.
    """
    df = pd.read_csv(path)
    print(f"Ma'lumot yuklandi: {len(df)} qator, {df.columns.tolist()[:5]}...")
    return df


def split_features_target(df: pd.DataFrame):
    """X (belgilar) va y (churn yorlig'i) ga ajratadi.

    Meta ustunlar (customer_id, timestamps) ham X dan olib tashlanadi —
    ular bashorat uchun belgi emas, shunchaki identifikator.

    Returns:
        X (DataFrame): 19 ta belgi
        y (Series): 0/1 yorliqlar
    """
    # Olib tashlanadigan ustunlar: meta + yorliq
    drop_cols = [c for c in META_COLUMNS + [TARGET] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET]
    return X, y


def build_pipeline() -> Pipeline:
    """sklearn Pipeline quramiz: OneHotEncoder (kategorik) + passthrough (raqamli) + RandomForest.

    Nima uchun aniq ustun ro'yxati?
      - Train va serve vaqtida bir xil ustun tartibi kafolatlanadi
      - Yangi kategoriya kelsa (handle_unknown="ignore") — xato emas, 0 qo'yiladi
      - class_weight="balanced": Churn=Yes atigi ~26% — kamchilikka ko'proq og'irlik beradi

    Returns:
        Hali o'qitilmagan Pipeline (fit() chaqirilishi kerak)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,           # barcha CPU yadrolaridan foydalanadi
        class_weight="balanced",  # nomutanosib sinflarga moslashish
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_test_split_data(X: pd.DataFrame, y: pd.Series):
    """Stratifitsiyalangan train/test bo'linishi.

    stratify=y: train va test da churn nisbati bir xil saqlanadi (~26%).
    Bu ayniqsa kichik sinf (Churn=1) uchun muhim — tasodifiy bo'linishda
    test da kamroq yoki ko'proq churn tushib qolishi mumkin.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Test to'plamida metrikalarni hisoblaydi.

    Returns:
        metrics dict: MLflow.log_metrics() ga to'g'ridan-to'g'ri beriladi.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # "ketadi" ehtimoli

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # Konsolga chiqarish (Prefect log_prints=True bilan ushlab oladi)
    print("\n=== Metrics (test set) ===")
    for name, value in metrics.items():
        print(f"  {name:9}: {value:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

    return metrics
