#!/usr/bin/env bash
# EC2 smoke test script for the Churn MLOps API.
# MLOps step: Phase 5, low-cost cloud deployment smoke test before Kubernetes.

set -euo pipefail
# set -e stops on the first failed command.
# set -u fails when an undefined variable is used.
# set -o pipefail fails the script when any command in a pipeline fails.

APP_IMAGE="churn-api-smoke:latest"
# APP_IMAGE is the local Docker image name built on the EC2 instance.

CONTAINER_NAME="churn-api-smoke"
# CONTAINER_NAME is the container name used for start/stop cleanup.

PORT="8000"
# PORT is the public API port exposed from the Docker container.

AWS_REGION="${AWS_REGION:-ap-northeast-2}"
# AWS_REGION defaults to Seoul because the project DVC bucket is in ap-northeast-2.

if [[ ! -f "Dockerfile" || ! -f ".dvc/config" ]]; then
  # This guard makes sure the script is run from the repository root.
  echo "Run this script from the Churn_MLOps repository root."
  exit 1
fi

python3 -m venv .venv-smoke
# Create an isolated Python environment for DVC on the EC2 instance.

source .venv-smoke/bin/activate
# Activate the smoke-test virtual environment in the current shell.

python -m pip install --upgrade pip
# Upgrade pip so binary wheels install more reliably.

python -m pip install dvc==3.59.0 dvc-s3==3.3.0 pathspec==0.12.1
# Install only the DVC packages needed to pull artifacts from S3.

if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
  # Ask for the key only when it was not already set in the shell.
  read -r -p "AWS Access key ID: " AWS_ACCESS_KEY_ID
  export AWS_ACCESS_KEY_ID
fi

if [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  # Read the secret without printing it to the terminal.
  read -r -s -p "AWS Secret access key: " AWS_SECRET_ACCESS_KEY
  echo
  export AWS_SECRET_ACCESS_KEY
fi

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
# Tell boto3/s3fs which AWS region the S3 DVC bucket uses.

python -m dvc pull models/churn_model.joblib
# Download the serving model artifact from the configured S3 DVC remote.

unset AWS_ACCESS_KEY_ID
# Remove the access key from the current shell after DVC pull.

unset AWS_SECRET_ACCESS_KEY
# Remove the secret key from the current shell after DVC pull.

unset AWS_DEFAULT_REGION
# Remove the AWS region variable from the current shell after DVC pull.

sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
# Delete any old smoke-test container so the new run starts cleanly.

sudo docker build -t "$APP_IMAGE" .
# Build the API Docker image from the repository Dockerfile.

sudo docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:8000" \
  -e MODEL_SOURCE=local \
  -e LOCAL_MODEL_PATH=/app/models/churn_model.joblib \
  -e DATABASE_URL=sqlite:////tmp/churn-api-smoke.db \
  -v "$(pwd)/models:/app/models:ro" \
  "$APP_IMAGE"
# Run the API container with the DVC-pulled local model mounted read-only.

sleep 10
# Give Uvicorn enough time to start before HTTP checks.

curl -fsS "http://127.0.0.1:${PORT}/health"
# Verify that the API liveness endpoint is reachable from the EC2 instance.

curl -fsS "http://127.0.0.1:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
# Send one real prediction request through the Dockerized API.

curl -fsS "http://127.0.0.1:${PORT}/metrics" | grep -E "churn_api_http_requests_total|churn_api_predictions_total"
# Verify that Prometheus metrics are exposed and prediction metrics are recorded.

echo
# Print a blank line for readable output.

echo "Smoke test passed. Open http://<EC2_PUBLIC_IP>:8000/health from your browser if port 8000 is allowed."
# Tell the operator what external URL to test from their laptop.

echo "When finished, stop costs with: sudo docker rm -f $CONTAINER_NAME"
# Remind the operator how to stop the container on EC2.
