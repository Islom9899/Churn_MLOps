# Production Docker image for the Churn Prediction API.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/

ENV MODEL_SOURCE=registry
ENV MLFLOW_MODEL_ALIAS=production
ENV ALLOW_LOCAL_MODEL_FALLBACK=false

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
