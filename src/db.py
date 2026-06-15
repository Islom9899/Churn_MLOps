"""
db.py — PostgreSQL bazaga ulanish (SQLAlchemy engine + sessiya fabrikasi).

Ulanish manzili .env faylidagi DATABASE_URL dan olinadi.
Butun loyiha (ingestion, training) shu `engine` va `SessionLocal`ni qayta ishlatadi.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()  # .env faylni o'qiydi (DATABASE_URL shu yerda)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL topilmadi — .env faylga qo'sh.")

# engine = bazaga ulanish "ko'prigi". Bir marta yaratiladi, hamma joyda ishlatiladi.
engine = create_engine(DATABASE_URL)

# SessionLocal = sessiya "fabrikasi". Har bir DB amali (yozish/o'qish) shundan sessiya oladi.
SessionLocal = sessionmaker(bind=engine)


def check_connection() -> None:
    """Bazaga ulanib, versiyasini chiqaradi — ulanish ishlayotganini tekshirish uchun."""
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
    print("PostgreSQL ulandi:")
    print(" ", version)


if __name__ == "__main__":
    check_connection()