"""
validation.py — Pandera orqali ma'lumot validatsiyasi.

Ikkita sxema:
  customer_features_schema — 19 ta belgi uchun (trening va bashorat oldidan)
  training_data_schema     — PIT join natijasi uchun (belgilar + churn yorlig'i)

Ikkita tekshiruv funksiyasi:
  validate_features(df)  — belgilarni tekshiradi, xato bo'lsa SchemaError raise qiladi
  validate_training(df)  — trening ma'lumotini tekshiradi

Nima uchun funksional API (DataFrameSchema) ishlatilgan?
  pandera 0.21 + pandas 2.3.x da DataFrameModel ning Series[str] type annotation'i
  numpy dtype konversiyasida xatoga uchraydi (ma'lum bug). DataFrameSchema API
  yillardан barqaror ishlaydi va bir xil natija beradi.
"""

import pandera as pa
import pandas as pd

# ---- CSV -> DB ustun nomlari xaritalash ----
# CSV PascalCase ishlatadi, DB esa snake_case (PostgreSQL konvensiyasi).
CSV_TO_DB_COLUMNS: dict[str, str] = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
    # Quyidagilar allaqachon to'g'ri: gender, tenure
}


def rename_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CSV ustun nomlarini DB snake_case formatiga o'tkazadi.

    Faqat mavjud ustunlarni o'zgartiradi — yo'q ustunlarga e'tibor bermaydi.
    """
    return df.rename(columns=CSV_TO_DB_COLUMNS)


# ---- Ha/Yo'q va internet qo'shimchalari uchun qayta ishlatiladigan tekshiruvlar ----
_YES_NO = pa.Check.isin(["Yes", "No"])
_YES_NO_NO_PHONE = pa.Check.isin(["Yes", "No", "No phone service"])
_YES_NO_NO_INET = pa.Check.isin(["Yes", "No", "No internet service"])

# ---- Mijoz belgilari sxemasi (19 ta ustun) ----
# Har pa.Column uchun: tur, qiymat tekshiruvi, nullable=False (bo'sh bo'lmasin)
customer_features_schema = pa.DataFrameSchema(
    columns={
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"]), nullable=False),
        "senior_citizen": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "partner": pa.Column(str, _YES_NO, nullable=False),
        "dependents": pa.Column(str, _YES_NO, nullable=False),
        "tenure": pa.Column(  # xizmat muddati: 0..120 oy
            int,
            [pa.Check.greater_than_or_equal_to(0), pa.Check.less_than_or_equal_to(120)],
            nullable=False,
        ),
        "phone_service": pa.Column(str, _YES_NO, nullable=False),
        "multiple_lines": pa.Column(str, _YES_NO_NO_PHONE, nullable=False),
        "internet_service": pa.Column(
            str, pa.Check.isin(["DSL", "Fiber optic", "No"]), nullable=False
        ),
        "online_security": pa.Column(str, _YES_NO_NO_INET, nullable=False),
        "online_backup": pa.Column(str, _YES_NO_NO_INET, nullable=False),
        "device_protection": pa.Column(str, _YES_NO_NO_INET, nullable=False),
        "tech_support": pa.Column(str, _YES_NO_NO_INET, nullable=False),
        "streaming_tv": pa.Column(str, _YES_NO_NO_INET, nullable=False),
        "streaming_movies": pa.Column(str, _YES_NO_NO_INET, nullable=False),
        "contract": pa.Column(
            str,
            pa.Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
        ),
        "paperless_billing": pa.Column(str, _YES_NO, nullable=False),
        "payment_method": pa.Column(
            str,
            pa.Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
            nullable=False,
        ),
        "monthly_charges": pa.Column(  # oylik to'lov: 0..500 dollar
            float,
            [pa.Check.greater_than_or_equal_to(0.0), pa.Check.less_than_or_equal_to(500.0)],
            nullable=False,
        ),
        "total_charges": pa.Column(  # jami to'lov: manfiy bo'lmasin
            float,
            pa.Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
    },
    # coerce=True: tur konversiyasiga ruxsat. "42" -> 42 (int) kabi kichik farqlar xatoga olib kelmaydi.
    coerce=True,
    # strict=False: qo'shimcha ustunlarga ruxsat (customer_id, timestamps ham bo'lishi mumkin).
    strict=False,
)

# ---- Trening ma'lumoti sxemasi: 19 belgi + churn yorlig'i ----
# customer_features_schema ga asoslangan, faqat churn ustuni qo'shiladi.
training_data_schema = pa.DataFrameSchema(
    columns={
        **customer_features_schema.columns,  # barcha 19 belgi tekshiruvi meros
        "churn": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),  # yorliq
    },
    coerce=True,
    strict=False,
)


# ---- Tekshiruv funksiyalari ----

def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """19 ta belgi DataFrame ni customer_features_schema bilan tekshiradi.

    Returns:
        Tekshirilgan DataFrame (o'zgarmagan).
    Raises:
        pa.errors.SchemaError: xato bo'lsa — aniq qaysi ustun, qaysi qator ko'rinadi.
    """
    print(f"[Validatsiya] {len(df)} qator, {len(df.columns)} ustun tekshirilmoqda...")
    validated = customer_features_schema.validate(df)
    print("[Validatsiya] Belgilar sxemasi: OK")
    return validated


def validate_training(df: pd.DataFrame) -> pd.DataFrame:
    """Trening ma'lumotini training_data_schema bilan tekshiradi.

    Returns:
        Tekshirilgan DataFrame.
    Raises:
        pa.errors.SchemaError: xato bo'lsa.
    """
    print(f"[Validatsiya] Trening ma'lumoti: {len(df)} qator tekshirilmoqda...")
    validated = training_data_schema.validate(df)
    churn_rate = df["churn"].mean()
    print(f"[Validatsiya] Trening sxemasi: OK | Churn nisbati: {churn_rate:.1%}")
    return validated
