"""
train.py — Telco Customer Churn model training (Phase 1).

Pipeline:
    load CSV -> clean -> build sklearn Pipeline (preprocess + RandomForest)
    -> train/test split -> evaluate -> save the full pipeline.

Run:  python src/train.py
"""

from pathlib import Path

import joblib
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

# ---- Config -------------------------------------------------------------
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known data issues and prepare the target column."""
    df = df.copy()

    # customerID is an identifier, not a predictive feature.
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text and has 11 blank values
    # (brand-new customers, tenure = 0). Coerce to numeric -> blanks
    # become NaN, then fill with 0 (they have ~no charges yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Target: Yes/No -> 1/0
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    """One ColumnTransformer (one-hot for categoricals) + RandomForest."""
    features = df.drop(columns=[TARGET])
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = features.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",  # mild churn imbalance (~26.5% Yes)
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate(pipeline: Pipeline, X_test, y_test) -> None:
    """Print the key classification metrics on the held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Metrics (test set) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1       : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No", "Yes"]))


def main() -> None:
    df = clean_data(load_data(DATA_PATH))

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(df)
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
