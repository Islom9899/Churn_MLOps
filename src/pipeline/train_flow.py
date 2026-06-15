"""
train_flow.py — Prefect flow: churn modelini trening pipeline ni orchestratsiya qiladi.

Qadamlar (@task lar):
  1. load_and_validate  — CSV o'qiydi + Pandera validatsiya
  2. split_and_train    — features/target ajratadi, train/test bo'ladi, model o'qitadi
  3. evaluate_model     — metrikalarni hisoblaydi
  4. log_and_register   — MLflow ga yozadi + Model Registry ga ro'yxatga oladi + "staging" alias

Prefect nima uchun?
  Oddiy misol: Prefect — konveyer lenta. Har @task bitta stansiya.
  Bitta stansiya to'xtasa, lenta to'xtaydi (xato ushlanadi).
  UI da har stansiya rangi: yashil (muvaffaqiyat), qizil (xato), sariq (ishlayapti).

  Senior daraja: @task lar mustaqil qayta ishga tushishi mumkin (retry).
  @flow barcha @task larni bog'laydi, run tarixini saqlaydi, monitoring qiladi.
  Bu "python script" dan farqi: observability, retry, scheduling, alerting.

MLflow Model Registry:
  model versiyalari saqlanadi + "staging"/"production" alias bilan boshqariladi.
  Serve qiluvchi app (Phase 3) doim "production" alias dan modelni yuklab oladi.
  Yangi model avval "staging" ga tushadi — sifat tekshiruvidan keyin "production" ga o'tadi.

Ishga tushirish: python -m src.pipeline.train_flow
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split

from src.data.validation import validate_training
from src.train import (
    FEATURE_COLUMNS,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    TRAINING_DATA_PATH,
    build_pipeline,
    evaluate,
    load_training_data,
    split_features_target,
    train_test_split_data,
)

# ---- MLflow sozlamalari ----
TRACKING_URI = "sqlite:///mlflow.db"       # lokal sqlite baza (mlflow ui ko'radi)
EXPERIMENT_NAME = "churn-prediction"        # MLflow UI dagi eksperiment nomi
REGISTERED_MODEL_NAME = "churn-prediction"  # Registry dagi model nomi
STAGING_ALIAS = "staging"                   # yangi versiyaga beriladigan alias

# Dataset hash o'qiladigan DVC fayl
DATASET_DVC_FILE = Path("data/training_dataset.csv.dvc")

# Staging ga o'tish uchun minimal metrika chegaralari
# (Phase 4 da bu gate CI/CD da ishlatiladi: chegaradan past bo'lsa deploy bo'lmaydi)
MIN_ROC_AUC = 0.78
MIN_F1 = 0.50


def _read_dataset_hash(dvc_path: Path = DATASET_DVC_FILE) -> str:
    """DVC fayl ichidagi MD5 hash ni o'qiydi.

    Nima uchun: har MLflow run'ga dataset hash yoziladi.
    Bu model + dataset juftligini to'liq qayta tiklab olish imkonini beradi.
    "Shu model qaysi dataset bilan o'qitilgan?" savoliga javob shu hash.
    """
    with open(dvc_path) as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]


# ---- Prefect @task lar ----

@task(name="load-and-validate", retries=1)
def load_and_validate_task(path: Path) -> object:
    """CSV o'qiydi va Pandera bilan validatsiya qiladi.

    retries=1: agar disk xatosi bo'lsa, bir marta qayta urinadi.
    Validatsiya xatosi bo'lsa — qayta urinmaydi (ma'lumot muammosi, dastur to'xtaydi).
    """
    print(f"[1/4] Ma'lumot yuklanmoqda: {path}")
    df = load_training_data(path)
    validate_training(df)   # Pandera: sxema + diapason + kategoriya tekshiruvi
    print(f"      Yuklandi va validatsiya qilindi: {len(df)} qator")
    return df


@task(name="split-and-train")
def split_and_train_task(df: object) -> tuple:
    """Belgilar/yorliq ajratadi, train/test bo'ladi, modelni o'qitadi.

    Returns:
        (pipeline, X_test, y_test) — baholash uchun kerak
    """
    print("[2/4] Trening boshlanmoqda...")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"      Train: {len(X_train)} | Test: {len(X_test)} | Churn (train): {y_train.mean():.1%}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      Model o'qitildi.")
    return pipeline, X_test, y_test


@task(name="evaluate-model")
def evaluate_task(pipeline: object, X_test: object, y_test: object) -> dict:
    """Test to'plamida metrikalarni hisoblaydi."""
    print("[3/4] Baholash...")
    metrics = evaluate(pipeline, X_test, y_test)
    return metrics


@task(name="log-and-register")
def log_and_register_task(pipeline: object, metrics: dict, dataset_hash: str) -> int:
    """MLflow ga yozadi, Registry ga qo'shadi, 'staging' alias beradi.

    MLflow Model Registry:
      - Har trening = yangi model versiyasi (1, 2, 3, ...)
      - 'staging' alias = sifat tekshiruvini kutayotgan versiya
      - 'production' alias = hozir serve qilayotgan versiya (Phase 3 da o'rnatiladi)

    Returns:
        Ro'yxatga olingan model versiyasi raqami.
    """
    print("[4/4] MLflow ga yozilmoqda va Registry ga qo'shilmoqda...")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Parametrlar: qanday sozlamalar bilan o'qitildi
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "feature_count": len(FEATURE_COLUMNS),
        })

        # Metrikalar: natijalar
        mlflow.log_metrics(metrics)

        # Dataset tegi: QAYSI dataset bilan o'qitildi — reproducibility uchun muhim
        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("dataset_path", str(TRAINING_DATA_PATH))
        mlflow.set_tag("target_column", TARGET)

        # Modeli o'zi artefakt sifatida
        mlflow.sklearn.log_model(pipeline, name="model")

        run_id = run.info.run_id

    print(f"      MLflow run_id: {run_id}")

    # Model Registry ga ro'yxatga olamiz
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    version = int(result.version)

    # "staging" alias berish: bu versiya sifat tekshiruvini kutmoqda
    # mlflow 3.x da stage o'rniga alias ishlatiladi (stages deprecated)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=STAGING_ALIAS,
        version=str(version),
    )

    print(f"      Model Registry: '{REGISTERED_MODEL_NAME}' v{version} -> alias='{STAGING_ALIAS}'")
    print(f"      roc_auc={metrics['roc_auc']:.4f} | f1={metrics['f1']:.4f}")
    return version


# ---- Prefect @flow ----

@flow(name="churn-training-pipeline", log_prints=True)
def training_pipeline(data_path: Path = TRAINING_DATA_PATH) -> dict:
    """Churn modelini o'qitish pipeline si.

    Barcha qadamlar ketma-ket, har biri alohida Prefect task:
    yukla → validatsiya → trening → baholash → ro'yxatga ol.

    log_prints=True: barcha print() ni Prefect UI da log sifatida ko'rsatadi.

    Args:
        data_path: trening ma'lumoti CSV yo'li (odatda DVC dan keladi).

    Returns:
        {"version": int, "metrics": dict} — CI gate uchun ishlatiladi (Phase 4).
    """
    # Dataset hash: reproducibility uchun MLflow ga yoziladi
    dataset_hash = _read_dataset_hash()
    print(f"Dataset hash (DVC): {dataset_hash}")

    # 1. Yukla + validatsiya
    df = load_and_validate_task(data_path)

    # 2. Trening
    pipeline, X_test, y_test = split_and_train_task(df)

    # 3. Baholash
    metrics = evaluate_task(pipeline, X_test, y_test)

    # 4. MLflow + Registry
    version = log_and_register_task(pipeline, metrics, dataset_hash)

    print(f"\nPipeline yakunlandi: model v{version} '{STAGING_ALIAS}' aliasida.")
    return {"version": version, "metrics": metrics}


if __name__ == "__main__":
    # To'g'ridan-to'g'ri chaqirilganda:  python -m src.pipeline.train_flow
    result = training_pipeline()
    print(f"\nNatija: {result}")
