"""
prepare.py — Phase 1 data layer ni to'liq ishga tushiradi.

Ketma-ketlik (har qadam muvaffaqiyatli bo'lmasa keyingisi ishlamaydi):
  1. Yangi DB jadvallar yaratiladi: customer_features, churn_labels
  2. Xom CSV Pandera bilan validatsiya qilinadi (sxema + qiymat tekshiruvi)
  3. DB ga seed qilinadi (idempotent)
  4. Point-in-time join bajariladi
  5. Trening ma'lumoti validatsiya qilinadi
  6. data/training_dataset.csv ga eksport qilinadi

Keyin qo'lda bajariladigan:
  dvc add data/training_dataset.csv   <- DVC versiyalash
  git add + git commit                <- git tarixiga yozish

Ishga tushirish: python -m src.data.prepare
"""

import sys
from pathlib import Path

import pandas as pd
import pandera as pa

from src.models import init_db
from src.data.seed import seed_db, DATA_PATH, _load_and_clean_csv
from src.data.pit_join import export_training_dataset, FEATURE_COLUMNS
from src.data.validation import validate_features, validate_training


def prepare_dataset() -> None:
    """Data layer ni to'liq tayyorlaydi. Biror qadam muvaffaqiyatsiz bo'lsa — to'xtaydi."""

    print("=" * 60)
    print("PHASE 1: Ma'lumot qatlami tayyorlash")
    print("=" * 60)

    # --- 1. DB jadvallarini yaratish ---
    print("\n[1/5] DB jadvallari yaratilmoqda (customer_features, churn_labels)...")
    init_db()

    # --- 2. Xom CSV ni validatsiya qilish ---
    print("\n[2/5] Xom CSV validatsiya qilinmoqda (Pandera)...")
    try:
        raw_df = _load_and_clean_csv(DATA_PATH)
        # Faqat 19 ta belgi ustunini validatsiya qilamiz (customer_id va churn o'chiriladi)
        features_only = raw_df[FEATURE_COLUMNS]
        validate_features(features_only)
    except pa.errors.SchemaError as exc:
        # Sxema xatosi: aniq qaysi ustun, qaysi qator — barchasi ko'rinadi
        print(f"\n[XATO] Validatsiya muvaffaqiyatsiz:\n{exc}")
        sys.exit(1)  # 0 bo'lmagan exit code -> CI ga: "muvaffaqiyatsiz"

    # --- 3. DB ga seed qilish ---
    print("\n[3/5] DB ga seed qilinmoqda...")
    seed_db()

    # --- 4. Point-in-time join ---
    print("\n[4/5] Point-in-time join bajarilmoqda...")
    output_path = export_training_dataset()

    # --- 5. Trening ma'lumotini validatsiya qilish ---
    print("\n[5/5] Trening ma'lumoti validatsiya qilinmoqda...")
    try:
        train_df = pd.read_csv(output_path)
        validate_training(train_df)
    except pa.errors.SchemaError as exc:
        print(f"\n[XATO] Trening ma'lumoti validatsiyasi muvaffaqiyatsiz:\n{exc}")
        sys.exit(1)

    # --- Muvaffaqiyat ---
    print("\n" + "=" * 60)
    print(f"MUVAFFAQIYAT! Trening ma'lumoti tayyor: {output_path}")
    print()
    print("Keyingi qadamlar (qo'lda):")
    print("  dvc add data/training_dataset.csv")
    print("  git add data/training_dataset.csv.dvc data/.gitignore")
    print('  git commit -m "data: add versioned training dataset (Phase 1)"')
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()
