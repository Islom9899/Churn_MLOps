# ===== Churn Prediction API uchun Docker image =====
# Yengil rasmiy Python image (slim = kichik hajm, kamroq keraksiz narsa).
FROM python:3.13-slim

# Konteyner ichidagi ishchi papka (bundan keyin hamma narsa shu yerda).
WORKDIR /app

# 1) AVVAL faqat requirements nusxalanadi va o'rnatiladi.
#    Docker kesh hiylasi: kodingni o'zgartirsang ham, requirements o'zgarmasa,
#    paketlar QAYTA o'rnatilmaydi -> build ancha tez bo'ladi.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) Kerakli fayllarni nusxalaymiz: kod (src/) va tayyor model (models/).
#    Diqqat: build'dan oldin `python src/train.py` ishlatib model yaratilgan bo'lsin.
COPY src/ ./src/
COPY models/ ./models/

# 3) Konteyner qaysi portni "ochishini" bildiradi (hujjat uchun; pastdagi --port bilan mos).
EXPOSE 8000

# 4) Konteyner ishga tushganda bajariladigan buyruq.
#    --host 0.0.0.0 -> konteyner TASHQARISIDAN ham kirish mumkin (faqat localhost emas).
#    --reload YO'Q -> u faqat ishlab chiqish uchun, productionда kerak emas.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]