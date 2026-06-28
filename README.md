# Customer Churn Prediction — Production-Grade MLOps Pipeline

An end-to-end MLOps system that predicts telecom customer churn, built to production
engineering standards rather than as a notebook prototype. It covers the full lifecycle:
a versioned data layer with point-in-time correctness, an orchestrated training pipeline
with experiment tracking and a model registry, a REST serving API, and CI that retrains
and tests on every push.

The dataset is the public [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
set (7,043 customers, 19 features, ~26.5% churn rate).

---

## Why this project exists

Most churn demos stop at `model.fit()` in a notebook. The hard part of shipping ML is
everything around the model: keeping data reproducible, preventing label leakage,
versioning datasets and models together, automating retraining, and serving predictions
reliably. This repository is structured to demonstrate **those** engineering concerns.

Key design decisions:

- **Point-in-time (PIT) joins** to build the training set, so a model is never trained
  on information that wouldn't have been available at prediction time (no label leakage).
- **Features and labels stored in separate tables**, mirroring reality where the outcome
  (churn) is only known ~30 days after the features are observed.
- **Dataset + model versioned together** — every MLflow run records the exact DVC hash of
  the data it was trained on, so any model is fully reproducible.
- **Schema validation as a pipeline gate** (Pandera) — bad data fails the run instead of
  silently degrading the model.

---

## Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Telco CSV   │ ──▶ │  Neon PostgreSQL     │ ──▶ │  PIT join + Pandera  │
│  (DVC)       │     │  customer_features   │     │  → training_dataset  │
│              │     │  churn_labels        │     │    (DVC tracked)     │
└──────────────┘     └─────────────────────┘     └──────────┬───────────┘
                                                             │
                                                             ▼
                                          ┌──────────────────────────────────┐
                                          │  Prefect training flow            │
                                          │  load → validate → train →        │
                                          │  evaluate → register              │
                                          └──────────────┬────────────────────┘
                                                         │
                              ┌──────────────────────────┴───────────┐
                              ▼                                       ▼
                  ┌────────────────────────┐              ┌────────────────────┐
                  │  MLflow Tracking +     │              │  FastAPI service   │
                  │  Model Registry        │ ──(Phase 3)─▶│  /predict /ingest  │
                  │  (staging/production)  │              │  /health           │
                  └────────────────────────┘              └────────────────────┘
```

---

## Tech stack

| Concern               | Tool                                            |
| --------------------- | ----------------------------------------------- |
| Language              | Python 3.13                                      |
| Modeling              | scikit-learn (RandomForest, `class_weight=balanced`) |
| Orchestration         | Prefect 3                                        |
| Experiment tracking   | MLflow (Tracking + Model Registry)               |
| Data validation       | Pandera                                          |
| Data/model versioning | DVC                                              |
| Database              | PostgreSQL (Neon) via SQLAlchemy                 |
| Serving               | FastAPI + Uvicorn                                |
| Packaging             | Docker                                           |
| CI                    | GitHub Actions                                   |
| Testing               | pytest + httpx                                   |

---

## Repository layout

```
src/
├── db.py                  # SQLAlchemy engine + session factory (reads DATABASE_URL)
├── models.py              # ORM tables: customer_features, churn_labels (+ legacy customers)
├── data/
│   ├── seed.py            # Telco CSV → PostgreSQL (idempotent, deterministic timestamps)
│   ├── pit_join.py        # Point-in-time join → data/training_dataset.csv
│   ├── validation.py      # Pandera schema (functional API)
│   └── prepare.py         # Orchestrates seed → pit_join → validate
├── train.py               # Pure training functions (no orchestration, importable)
├── pipeline/
│   └── train_flow.py      # Prefect flow: load → validate → train → evaluate → register
└── app.py                 # FastAPI service: /health, /predict, /ingest
tests/                     # pytest suite (API + data layer)
.github/workflows/ci.yml   # install → prepare data → train → test on every push/PR
Dockerfile                 # Slim production image for the serving API
```

---

## Getting started

### Prerequisites

- Python 3.13
- A PostgreSQL database (this project uses [Neon](https://neon.tech))
- DVC (installed via `requirements.txt`)

### 1. Install

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash); use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### 2. Configure the database

```bash
cp .env.example .env
# edit .env and set DATABASE_URL=postgresql+psycopg2://<user>:<pass>@<host>/<db>
```

### 3. Prepare the training data

```bash
python -m src.data.prepare        # seed DB → PIT join → validate → training_dataset.csv
```

If the data is already versioned in a DVC remote, you can instead pull it:

```bash
dvc pull data/training_dataset.csv
```

### 4. Train the model

```bash
python -m src.pipeline.train_flow
```

This runs the Prefect flow, logs the run to MLflow, registers a new model version, and
assigns it the `staging` alias. Inspect runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

### 5. Serve predictions

```bash
uvicorn src.app:app --reload
# Swagger docs at http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,
       "PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL",
       "OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":29.85,"TotalCharges":29.85}'
```

---

## API endpoints

| Method | Path       | Description                                                |
| ------ | ---------- | ---------------------------------------------------------- |
| GET    | `/health`  | Liveness check.                                            |
| POST   | `/predict` | Returns churn prediction + probability for one customer.   |
| POST   | `/ingest`  | Persists a labeled customer record (features + actual churn) to the database for future retraining. |

---

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request
to `main`/`master`:

1. Install dependencies
2. Prepare the training data (`dvc pull`, falling back to a full rebuild via `prepare.py`)
3. Train the model through the Prefect flow and register it in MLflow
4. Run the pytest suite

`DATABASE_URL` is provided as a GitHub Actions secret.

---

## Quality gates

The training flow defines minimum metric thresholds (`MIN_ROC_AUC = 0.78`, `MIN_F1 = 0.50`).
A model below these is not a deploy candidate — in Phase 4 this becomes an automated
promotion gate (staging → production) in CI/CD.

---

## Roadmap

| Phase | Scope                                                      | Status        |
| ----- | --------------------------------------------------------- | ------------- |
| 1     | Data layer: PostgreSQL schema, PIT join, Pandera, DVC      | ✅ Done        |
| 2     | Training pipeline: Prefect flow, MLflow tracking + registry | ✅ Done        |
| 3     | Serving from the registry (`production` alias), retire legacy table | 🔜 Next |
| 4     | Automated promotion gate, monitoring, drift detection      | 📋 Planned    |

> **Note:** This README is living documentation and is updated as each phase lands. The
> serving API currently loads a local `models/churn_model.joblib` artifact; Phase 3 switches
> it to pull the `production`-aliased model directly from the MLflow Model Registry.
