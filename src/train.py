"""
train.py — Telco Customer Churn modelini o'qitish (Phase 1 + MLflow tracking).

Bu fayl nima qiladi (ketma-ketlik):
    CSV o'qish -> tozalash -> Pipeline (preprocess + RandomForest) qurish
    -> train/test ga bo'lish -> o'qitish -> baholash -> MLflow'ga yozish -> modelni saqlash.

Ishga tushirish:   python src/train.py
Natijani ko'rish:  mlflow ui --backend-store-uri sqlite:///mlflow.db
                   (keyin brauzerda http://127.0.0.1:5000 oching)
"""

from pathlib import Path  # fayl yo'llari bilan toza ishlash (Windows/Linux farqisiz)

import joblib            # tayyor modelni diskka saqlash/yuklash uchun (.joblib fayl)
import mlflow            # eksperiment kuzatuvi: parametr / metrika / modelni qayd qilish
import mlflow.sklearn    # sklearn modelini MLflow formatida saqlash uchun
import pandas as pd      # ma'lumotni jadval (DataFrame) ko'rinishida ishlatish

# sklearn — ma'lumotni tayyorlash va model qismlari:
from sklearn.compose import ColumnTransformer        # turli ustunlarga turlicha ishlov berish
from sklearn.ensemble import RandomForestClassifier  # asosiy model (ko'p daraxtli "o'rmon")
from sklearn.metrics import (                         # baholash o'lchovlari
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # ma'lumotni train/test ga bo'lish
from sklearn.pipeline import Pipeline                 # bosqichlarni bitta zanjirga ulash
from sklearn.preprocessing import OneHotEncoder       # matnli kategoriyalarni raqamga aylantirish


# ---- Config: barcha sozlamalar bir joyda --------------------------------------
# "Sehrli raqam"larni kod ichiga sochmaymiz — bu yerda turadi, o'zgartirish oson.
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")       # kirish ma'lumoti (CSV) qayerda
MODEL_PATH = Path("models/churn_model.joblib")  # tayyor modelni qayerga saqlaymiz
TRACKING_URI = "sqlite:///mlflow.db"            # MLflow ma'lumotni shu sqlite bazaga yozadi
EXPERIMENT = "churn-prediction"                 # MLflow'dagi eksperiment nomi
TARGET = "Churn"                                # bashorat qilinadigan ustun (javob)
RANDOM_STATE = 42                               # tasodifni "qotirish" — har safar bir xil natija
TEST_SIZE = 0.2                                 # ma'lumotning 20% i test uchun ajraladi
N_ESTIMATORS = 200                              # o'rmondagi daraxtlar soni


def load_data(path: Path) -> pd.DataFrame:
    """CSV faylni o'qib, DataFrame (jadval) qaytaradi."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ma'lumotdagi ma'lum muammolarni tuzatadi va javob ustunini tayyorlaydi."""
    df = df.copy()  # asl jadvalni buzmaslik uchun nusxa ustida ishlaymiz

    # customerID — har mijozda noyob ID. Modelga foydasi yo'q (faqat yodlab oladi), olib tashlaymiz.
    df = df.drop(columns=["customerID"])

    # TotalCharges aslida matn bo'lib kelgan va 11 ta katak bo'sh (tenure=0 yangi mijozlar).
    #   to_numeric -> raqamga aylantiradi; aylanmaganini errors="coerce" NaN qiladi;
    #   fillna(0) -> bo'shlarni 0 ga to'ldiradi (yangi mijoz hali to'lamagan = 0, mantiqan to'g'ri).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Javob "Yes"/"No" matn — modelga raqam kerak: Yes -> 1, No -> 0.
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    """Preprocess (kategoriyalarni one-hot) + RandomForest ni bitta zanjirga (Pipeline) ulaydi."""
    features = df.drop(columns=[TARGET])  # javobdan boshqa hamma ustun = belgilar (X)

    # Ustunlarni 2 turga ajratamiz: matnli (kategorik) va raqamli.
    categorical = features.select_dtypes(include="object").columns.tolist()  # masalan: gender, Contract
    numeric = features.select_dtypes(exclude="object").columns.tolist()      # masalan: tenure, MonthlyCharges

    # ColumnTransformer — har turga boshqacha ishlov:
    #   cat -> OneHotEncoder: har bir variantni alohida 0/1 ustunga aylantiradi.
    #          handle_unknown="ignore": test'da ko'rilmagan yangi variant chiqsa, xato bermaydi.
    #   num -> "passthrough": raqamli ustunlar o'zgarmasdan o'tadi.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    # Asosiy model — RandomForest: ko'p qaror daraxti "ovoz beradi", ko'pchilik qaroriga ko'ra bashorat.
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,    # daraxtlar soni (ko'p = barqaror, lekin sekinroq)
        random_state=RANDOM_STATE,    # natija takrorlanishi uchun
        n_jobs=-1,                    # barcha CPU yadrolarini ishlatib tezlashtiradi
        class_weight="balanced",      # MUHIM: Churn=Yes atigi ~26% (kam sinf). Bu sozlama kam sinfga
                                      # ko'proq "og'irlik" beradi; aks holda model hammaga "No" deb yuborardi.
    )

    # Pipeline: avval preprocess, keyin model — birga fit/predict bo'ladi.
    # Foydasi: test va kelajakdagi yangi ma'lumotga ham AYNAN shu tozalash avtomat qo'llanadi (xato kamayadi).
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    """Metrikalarni hisoblaydi, ekranga chiqaradi va MLflow uchun dict qaytaradi."""
    y_pred = pipeline.predict(X_test)               # 0/1 bashorat (ketadi / ketmaydi)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # "ketadi" ehtimoli (0..1) — roc_auc uchun kerak

    metrics = {
        # accuracy: umumiy to'g'ri bashorat ulushi (nomutanosib ma'lumotda yolg'iz o'zi aldamchi bo'lishi mumkin).
        "accuracy": accuracy_score(y_test, y_pred),
        # precision: "ketadi" deganlarimdan nechtasi ROSTAN ketdi (yolg'on signal kamligi).
        "precision": precision_score(y_test, y_pred),
        # recall: rostan ketganlardan nechtasini USHLAY oldim (o'tkazib yuborish kamligi).
        "recall": recall_score(y_test, y_pred),
        # f1: precision va recall o'rtasidagi muvozanat (ikkalasini birga baholaydi).
        "f1": f1_score(y_test, y_pred),
        # roc_auc: model ehtimollarni qanchalik to'g'ri TARTIBLAYDI (0.5=tasodif, 1.0=mukammal).
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n=== Metrics (test set) ===")
    for name, value in metrics.items():
        print(f"{name:9}: {value:.3f}")
    # classification_report: har sinf (No/Yes) bo'yicha precision/recall/f1 jadvali.
    print("\n" + classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    return metrics


def main() -> None:
    # 1) Ma'lumotni o'qib, tozalaymiz.
    df = clean_data(load_data(DATA_PATH))

    # 2) X (belgilar) va y (javob) ga ajratamiz.
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 3) Train/test ga bo'lamiz.
    #    stratify=y -> train va test'da Yes/No nisbati bir xil saqlanadi (kam sinf yo'qolib ketmaydi).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4) MLflow sozlash: QAYERGA yozishni (sqlite baza) va qaysi eksperiment nomini belgilaymiz.
    #    Aniq belgilash muhim — shunda `mlflow ui --backend-store-uri sqlite:///mlflow.db` ayni shu run'ni ko'radi.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # 5) Bitta "run" (tajriba urinishi) ochamiz — ichidagi hamma narsa shu run'ga yoziladi.
    with mlflow.start_run():
        pipeline = build_pipeline(df)   # zanjirni quramiz
        pipeline.fit(X_train, y_train)  # o'qitamiz (faqat train ustida)

        metrics = evaluate(pipeline, X_test, y_test)  # test ustida baholaymiz

        # --- MLflow'ga qayd: parametrlar + metrikalar + modelning o'zi ---
        mlflow.log_params(  # qanday sozlamalar bilan o'qitganimiz (run'larni keyin solishtirish uchun)
            {
                "model": "RandomForest",
                "n_estimators": N_ESTIMATORS,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
            }
        )
        mlflow.log_metrics(metrics)                       # natijalar (accuracy, f1, roc_auc, ...)
        mlflow.sklearn.log_model(pipeline, name="model")  # modelning o'zi (artefakt sifatida)

        # 6) Modelni alohida fayl qilib ham saqlaymiz (FastAPI keyin AYNAN shu faylni yuklab ishlatadi).
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # models/ papkasi yo'q bo'lsa, yaratadi
        joblib.dump(pipeline, MODEL_PATH)
        print(f"\nModel saved -> {MODEL_PATH}")
        print("MLflow run logged. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to view it.")


if __name__ == "__main__":  # bu fayl to'g'ridan-to'g'ri ishga tushsa, main() chaqiriladi
    main()