"""
models.py — Bazadagi barcha jadvallar (SQLAlchemy ORM).

Jadvallar:
  customer_features  — mijoz belgilari (vaqt T dagi snapshot) [Phase 1 yangi]
  churn_labels       — churn natijasi + label_timestamp (T+30 kun) [Phase 1 yangi]
  customers          — LEGACY: eski bitta jadval (Phase 3 da o'chiriladi)

Arxitektura sababi:
  Belgilar va natijani ajratish — data leakage dan himoya.
  Real tizimda natija (Churn) belgilardan 30 kun KEYIN ma'lum bo'ladi.
  Shuning uchun ularni alohida jadvalda saqlash kerak.
"""

from sqlalchemy import Column, DateTime, Float, Integer, String, func
from sqlalchemy.orm import declarative_base

from src.db import engine

Base = declarative_base()  # barcha ORM modellari meros oladigan asos sinf


class CustomerFeature(Base):
    """Bitta mijozning belgilari snapshot'i — vaqt T dagi holat.

    snake_case ustun nomlari: PostgreSQL konvensiyasi, quoting muammolaridan xoli.
    customer_id bo'yicha index: PIT join da tez qidirish uchun.

    Haqiqiy tizimda: bir mijoz vaqt o'tishi bilan ko'p snapshot berishi mumkin
    (har oyda yangi snapshot). PIT join to'g'ri snapshotni tanlaydi.
    """

    __tablename__ = "customer_features"

    id = Column(Integer, primary_key=True)  # avtomat o'sadigan raqam
    customer_id = Column(String, nullable=False, index=True)  # mijoz identifikatori
    feature_timestamp = Column(  # belgilar qayd etilgan vaqt (T)
        DateTime(timezone=True), nullable=False
    )
    created_at = Column(  # baza yozuvini yaratgan vaqt (avtomat)
        DateTime(timezone=True), server_default=func.now()
    )

    # ---- 19 ta belgi (CSV ustunlaridan snake_case ga o'tkazilgan) ----
    gender = Column(String)           # jins: "Male" / "Female"
    senior_citizen = Column(Integer)  # katta yoshli: 0 yoki 1
    partner = Column(String)          # juftlik: "Yes" / "No"
    dependents = Column(String)       # qarindoshlar: "Yes" / "No"
    tenure = Column(Integer)          # xizmat muddati (oy)
    phone_service = Column(String)    # telefon xizmati bor/yo'q
    multiple_lines = Column(String)   # ko'p liniya
    internet_service = Column(String) # internet turi: "DSL" / "Fiber optic" / "No"
    online_security = Column(String)  # onlayn xavfsizlik
    online_backup = Column(String)    # onlayn zaxira
    device_protection = Column(String)# qurilma himoyasi
    tech_support = Column(String)     # texnik yordam
    streaming_tv = Column(String)     # TV oqimi
    streaming_movies = Column(String) # film oqimi
    contract = Column(String)         # shartnoma turi: oylik / 1 yil / 2 yil
    paperless_billing = Column(String)# elektron to'lov: "Yes" / "No"
    payment_method = Column(String)   # to'lov usuli
    monthly_charges = Column(Float)   # oylik to'lov (dollar)
    total_charges = Column(Float)     # jami to'lov (dollar)


class ChurnLabel(Base):
    """Mijozning churn natijasi — feature_timestamp dan KEYIN ma'lum bo'ladi.

    label_timestamp = feature_timestamp + CHURN_WINDOW (30 kun).

    Bu jadval train data to'plashda data leakage dan himoya qiladi:
    model faqat label_timestamp <= cutoff_date bo'lgan yozuvlar ustida o'qitiladi.
    Ya'ni: "kelajakdan" ma'lumot olish mumkin emas.
    """

    __tablename__ = "churn_labels"

    id = Column(Integer, primary_key=True)
    customer_id = Column(String, nullable=False, index=True)  # customer_features bilan bog'lanish
    churn = Column(Integer, nullable=False)  # 0 = qoldi, 1 = ketdi
    label_timestamp = Column(  # natija ma'lum bo'lgan vaqt (T + 30 kun)
        DateTime(timezone=True), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Customer(Base):
    """LEGACY jadval — Phase 3 da o'chiriladi.

    Hozircha app.py /ingest endpoint shu modeli ishlatadi.
    Phase 3 da CustomerFeature + ChurnLabel ga ko'chiriladi.
    """

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    gender = Column(String)
    SeniorCitizen = Column(Integer)
    Partner = Column(String)
    Dependents = Column(String)
    tenure = Column(Integer)
    PhoneService = Column(String)
    MultipleLines = Column(String)
    InternetService = Column(String)
    OnlineSecurity = Column(String)
    OnlineBackup = Column(String)
    DeviceProtection = Column(String)
    TechSupport = Column(String)
    StreamingTV = Column(String)
    StreamingMovies = Column(String)
    Contract = Column(String)
    PaperlessBilling = Column(String)
    PaymentMethod = Column(String)
    MonthlyCharges = Column(Float)
    TotalCharges = Column(Float)
    Churn = Column(String)


def init_db() -> None:
    """Barcha jadvallarni yaratadi (agar yo'q bo'lsa). Idempotent."""
    Base.metadata.create_all(engine)
    print("Jadvallar tayyor: customer_features, churn_labels, customers")


if __name__ == "__main__":
    init_db()
