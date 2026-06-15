"""
pit_join.py — Point-in-time (PIT) join: customer_features + churn_labels -> trening ma'lumoti.

Nima qiladi:
  1) customer_features jadvalidan barcha snapshotlarni o'qiydi
  2) churn_labels dan faqat cutoff_date gacha ma'lum bo'lgan natijalarni o'qiydi
  3) customer_id bo'yicha merge qiladi
  4) Leakage filtri: feature_timestamp < label_timestamp (belgilar natijadan oldin)
  5) Har mijoz uchun eng so'nggi snapshotni tanlaydi (DISTINCT ON ekvivalenti)
  6) data/training_dataset.csv ga eksport qiladi (DVC versiyalaydi)

Point-in-time join nima?
  Oddiy misol: Bankdan kredit so'rayapsiz. Bank SIZNING BUGUNGI holatingizdagi
  ma'lumotni ishlatadi, kelajakdagi ma'lumotni emas. Agar bank "siz keyinchalik
  qancha daromad olasiz" degan ma'lumotni ishlatsa — bu aldamchilik (leakage).
  PIT join: "faqat shu sanagacha ma'lum bo'lgan ma'lumotlarni ol" degan qoida.

  Senior daraja: ML modelini o'qitishda eng keng tarqalgan xato — label_timestamp
  dan oldin paydo bo'ladigan ma'lumotlarni belgi sifatida ishlatish. Bu modelning
  test'da yaxshi, production'da yomon ishlashiga olib keladi (training-serving skew).

Ishga tushirish: python -m src.data.pit_join
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.db import engine

# Trening uchun qirqim sanasi (cutoff date).
# Simulyatsiyamizda barcha snapshotlar 2023-01-01..2023-12-31 oralig'ida.
# label_timestamp = feature_timestamp + 30 kun, demak max label = 2023-12-31 + 30 = 2024-01-30.
# Biz 2024-02-01 qirqimini ishlatamiz — barcha 7043 mijoz tushadi.
DEFAULT_CUTOFF = datetime(2024, 2, 1, tzinfo=timezone.utc)

# Trening ma'lumoti CSV yo'li (DVC bu faylni kuzatadi)
OUTPUT_PATH = Path("data/training_dataset.csv")

# Trening uchun ishlatadigan 19 ta belgi ustunlari (tartib muhim: train.py da ishlatiladi)
FEATURE_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure",
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges",
]


def get_training_data(cutoff_date: datetime = DEFAULT_CUTOFF) -> pd.DataFrame:
    """Point-in-time join natijasini DataFrame qilib qaytaradi.

    Qadamlar:
      - customer_features o'qish
      - churn_labels o'qish (faqat cutoff_date gacha ma'lumlar)
      - Merge + leakage filtri + deduplikatsiya

    Args:
        cutoff_date: shu sanagacha label_timestamp bo'lgan natijalar ishlatiladi.

    Returns:
        DataFrame: 19 ta belgi + churn + meta ustunlar (customer_id, timestamps).
    """
    print(f"[PIT Join] Boshlandi (cutoff: {cutoff_date.date()})...")

    with engine.connect() as conn:
        # Barcha feature snapshotlarini o'qiymiz
        features_df = pd.read_sql(
            "SELECT * FROM customer_features", conn
        )

        # Faqat cutoff_date gacha natijasi ma'lum bo'lgan yozuvlarni o'qiymiz.
        # Bu yerda data leakage imkoni yo'q — kelajakdagi natijalar filtrlanadi.
        labels_df = pd.read_sql(
            "SELECT customer_id, churn, label_timestamp FROM churn_labels"
            " WHERE label_timestamp <= %(cutoff)s",
            conn,
            params={"cutoff": cutoff_date},
        )

    print(f"[PIT Join] Features: {len(features_df)} qator | Labels: {len(labels_df)} qator")

    # customer_id bo'yicha inner join (ikkalasida ham bo'lgan mijozlar)
    merged = features_df.merge(labels_df, on="customer_id", how="inner")

    # Leakage tekshiruvi: belgilar ALBATTA natijadan oldin bo'lishi kerak
    # feature_timestamp < label_timestamp — bu shartni kodni o'zi ta'minlaydi
    merged = merged[merged["feature_timestamp"] < merged["label_timestamp"]]

    # Har bir mijoz uchun eng so'nggi feature snapshotni tanlaymiz
    # (Hozirgi ma'lumotda har mijozda bitta snapshot, lekin kodimiz ko'pini ham qo'llab-quvvatlaydi)
    merged = (
        merged.sort_values("feature_timestamp", ascending=False)
        .drop_duplicates(subset=["customer_id"], keep="first")
        .reset_index(drop=True)
    )

    # Churn nisbatini tekshiramiz: dataset to'g'ri ekanligini vizual tasdiqlash
    churn_rate = merged["churn"].mean()
    print(
        f"[PIT Join] Natija: {len(merged)} mijoz | "
        f"Churn nisbati: {churn_rate:.1%} (kutilgan: ~26%)"
    )

    return merged


def export_training_dataset(
    cutoff_date: datetime = DEFAULT_CUTOFF,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """PIT join natijasini CSV ga yozadi — DVC versiyalash uchun.

    Eksport qilingan fayl keyingi bosqichda `dvc add data/training_dataset.csv`
    bilan versiyalanadi. Bu model + dataset juftligini git tarixi orqali tiklab
    olish imkonini beradi.

    Returns:
        Saqlangan fayl yo'li.
    """
    df = get_training_data(cutoff_date)

    # Faqat kerakli ustunlarni saqlаymiz: belgilar + yorliq + meta
    save_columns = ["customer_id", "feature_timestamp"] + FEATURE_COLUMNS + ["churn", "label_timestamp"]
    df_to_save = df[save_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_csv(output_path, index=False)
    print(f"[PIT Join] Saqlandi: {output_path} ({len(df_to_save)} qator)")
    return output_path


if __name__ == "__main__":
    export_training_dataset()
