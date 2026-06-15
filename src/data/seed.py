"""
seed.py — Telco CSV'dan customer_features va churn_labels jadvallarini to'ldiradi.

Nima qiladi (ketma-ketlik):
  1) CSV o'qiydi, TotalCharges tozalaydi
  2) Har mijoz uchun deterministik feature_timestamp hisoblaydi
     (customer_id MD5 hash asosida — har safar bir xil natija)
  3) customer_features + churn_labels jadvallariga bulk insert qiladi
  4) Idempotent: agar jadvalda yozuv bo'lsa, qayta qo'shmaydi

Nima uchun simulyatsiya?
  Telco CSV statik snapshot — haqiqiy vaqt tamg'alari yo'q.
  Real tizimda: har oyda yangi snapshot olinadi va haqiqiy timestamp bo'ladi.
  Biz deterministik tarzda vaqtlarni simulyatsiya qilamiz — bu takroriy ishlarda
  bir xil natija beradi (reproducibility).

Ishga tushirish: python -m src.data.seed
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.db import SessionLocal
from src.models import CustomerFeature, ChurnLabel
from src.data.validation import rename_csv_columns

# ---- Sozlamalar ----
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Simulyatsiya parametrlari
BASE_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)  # birinchi snapshot sanasi
YEAR_SPAN_DAYS = 365        # snapshotlar shu kun oralig'ida tarqaladi (2023 yil ichida)
CHURN_WINDOW_DAYS = 30      # churn kuzatish oynasi: natija 30 kundan keyin ma'lum

# Batch hajmi: bulk insertni kichik qismlarga bo'lish (xotira tejash uchun)
BATCH_SIZE = 500


def _compute_feature_timestamp(customer_id: str) -> datetime:
    """customer_id dan deterministik feature_timestamp hisoblaydi.

    Nima uchun MD5 hash?
      hash(customer_id) -> bir xil customer uchun har doim bir xil kun.
      Turli customerlar uchun turli kunlar (random ko'rinish, lekin takroriy).
      Real tizimda: haqiqiy timestamp bo'ladi, bu funksiya kerak bo'lmaydi.
    """
    hash_int = int(hashlib.md5(customer_id.encode()).hexdigest(), 16)
    day_offset = hash_int % YEAR_SPAN_DAYS  # 0..364 kun
    return BASE_DATE + timedelta(days=day_offset)


def _load_and_clean_csv(path: Path) -> pd.DataFrame:
    """CSV ni o'qiydi, TotalCharges tozalaydi, ustun nomlarini snake_case ga o'tkazadi."""
    df = pd.read_csv(path)

    # TotalCharges matn bo'lib kelgan: " " (bo'sh) -> NaN -> 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # CSV PascalCase -> DB snake_case (customer_id, senior_citizen, ...)
    df = rename_csv_columns(df)
    return df


def seed_db(csv_path: Path = DATA_PATH) -> None:
    """Neon PostgreSQL ga customer_features va churn_labels yozuvlarini qo'shadi.

    Idempotent: customer_features jadvalida yozuv bo'lsa, o'tkazib yuboradi.
    Bulk insert: bitta tranzaksiya, tezkor yozish.
    """
    print(f"[Seed] CSV o'qilmoqda: {csv_path}")
    df = _load_and_clean_csv(csv_path)
    print(f"[Seed] {len(df)} ta mijoz topildi.")

    with SessionLocal() as session:
        # Idempotentlik tekshiruvi — ikki marta seed qilishdan himoya
        existing_count = session.query(CustomerFeature).count()
        if existing_count > 0:
            print(
                f"[Seed] Jadvalda allaqachon {existing_count} ta yozuv bor."
                " Seeding o'tkazib yuborildi (idempotent)."
            )
            return

        features_batch: list[CustomerFeature] = []
        labels_batch: list[ChurnLabel] = []

        for _, row in df.iterrows():
            cid = row["customer_id"]

            # feature_timestamp: belgilar shu vaqtda qayd etildi
            feature_ts = _compute_feature_timestamp(cid)

            # label_timestamp: natija shu vaqtda ma'lum bo'ldi (kechikish!)
            # label_ts > feature_ts — bu gap data leakage dan himoya qiladi
            label_ts = feature_ts + timedelta(days=CHURN_WINDOW_DAYS)

            # 19 ta belgi snapshotini yozamiz
            features_batch.append(
                CustomerFeature(
                    customer_id=cid,
                    feature_timestamp=feature_ts,
                    gender=row["gender"],
                    senior_citizen=int(row["senior_citizen"]),
                    partner=row["partner"],
                    dependents=row["dependents"],
                    tenure=int(row["tenure"]),
                    phone_service=row["phone_service"],
                    multiple_lines=row["multiple_lines"],
                    internet_service=row["internet_service"],
                    online_security=row["online_security"],
                    online_backup=row["online_backup"],
                    device_protection=row["device_protection"],
                    tech_support=row["tech_support"],
                    streaming_tv=row["streaming_tv"],
                    streaming_movies=row["streaming_movies"],
                    contract=row["contract"],
                    paperless_billing=row["paperless_billing"],
                    payment_method=row["payment_method"],
                    monthly_charges=float(row["monthly_charges"]),
                    total_charges=float(row["total_charges"]),
                )
            )

            # Churn natijasini ALOHIDA jadvalda, kechiktirilgan vaqt bilan yozamiz
            labels_batch.append(
                ChurnLabel(
                    customer_id=cid,
                    churn=1 if str(row["churn"]).strip().lower() in ("yes", "1") else 0,
                    label_timestamp=label_ts,
                )
            )

            # BATCH_SIZE ga yetganda DBga yozamiz (xotirani tejash)
            if len(features_batch) >= BATCH_SIZE:
                session.bulk_save_objects(features_batch)
                session.bulk_save_objects(labels_batch)
                session.flush()  # tranzaksiyani yopmasdan DB ga yuboradi
                features_batch.clear()
                labels_batch.clear()

        # Qolgan yozuvlarni yozamiz
        if features_batch:
            session.bulk_save_objects(features_batch)
            session.bulk_save_objects(labels_batch)

        session.commit()  # barcha o'zgarishlarni DBga qotiradi

    print(f"[Seed] Muvaffaqiyatli: {len(df)} customer_features yozildi.")
    print(f"[Seed]                 {len(df)} churn_labels yozildi.")


if __name__ == "__main__":
    seed_db()
