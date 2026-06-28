# CI/CD deployment guide

This document describes the Phase 6 path: cloud artifact storage, container
registry publishing, and Kubernetes deployment automation.

## What was added

- `.github/workflows/container.yml` builds the FastAPI Docker image and pushes it to GHCR.
- `.github/workflows/deploy-k8s.yml` deploys a selected GHCR image tag to Kubernetes.
- `.github/workflows/ci.yml` can override the local DVC remote through GitHub Secrets.

## GitHub Actions secrets

Required for CI training:

| Secret | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string for training data preparation and ingestion tests. |
| `DVC_REMOTE_URL` | CI-accessible DVC remote URL. Example: S3, GCS, Azure Blob, WebDAV, or another shared remote. |

Optional for S3-compatible DVC remotes:

| Secret | Purpose |
| --- | --- |
| `DVC_AWS_ACCESS_KEY_ID` | Access key for S3-compatible artifact storage. |
| `DVC_AWS_SECRET_ACCESS_KEY` | Secret key for S3-compatible artifact storage. |
| `DVC_AWS_DEFAULT_REGION` | Region for S3-compatible artifact storage. |

Required for Kubernetes deploy:

| Secret | Purpose |
| --- | --- |
| `KUBE_CONFIG_B64` | Base64-encoded kubeconfig for the target cluster. |

Required inside Kubernetes:

| Kubernetes secret | Key | Purpose |
| --- | --- | --- |
| `churn-api-secrets` | `DATABASE_URL` | API database connection. |
| `churn-api-secrets` | `MLFLOW_TRACKING_URI` | MLflow tracking server used by the serving API. |

Create the Kubernetes secret once:

```bash
kubectl apply -f k8s/base/namespace.yaml
kubectl -n churn-mlops create secret generic churn-api-secrets \
  --from-literal=DATABASE_URL="postgresql+psycopg2://..." \
  --from-literal=MLFLOW_TRACKING_URI="https://..."
```

## Current S3 DVC remote

The project is configured for this AWS S3 remote:

```text
s3://churn-mlops-dvc-mansurov-ap-northeast-2/dvc
```

GitHub Secrets should contain:

```text
DVC_REMOTE_URL=s3://churn-mlops-dvc-mansurov-ap-northeast-2/dvc
DVC_AWS_DEFAULT_REGION=ap-northeast-2
```

For the first upload from your laptop, run:

```powershell
.\scripts\push_dvc_s3.ps1
```

The script asks for AWS keys at runtime and clears them from the process
environment after `dvc push`. Do not paste the keys into chat or commit them.

## Workflow order

1. `CI`
   - Installs Python dependencies.
   - Optionally overrides the DVC remote with `DVC_REMOTE_URL`.
   - Pulls or rebuilds training data.
   - Trains and promotes the model.
   - Runs tests.

2. `Container`
   - Builds the Docker image.
   - Publishes tags to `ghcr.io/islom9899/churn-api`.
   - Produces `latest`, branch, tag, and commit SHA tags.

3. `Deploy Kubernetes`
   - Runs manually from GitHub Actions.
   - Uses `KUBE_CONFIG_B64`.
   - Applies Kubernetes manifests.
   - Updates the deployment image to the selected tag.
   - Waits for rollout completion.

## DVC remote recommendation

The repo still keeps `local_remote` in `.dvc/config` for laptop development. In
GitHub Actions, the remote URL is overridden locally at runtime:

```bash
dvc remote modify --local local_remote url "$DVC_REMOTE_URL"
```

For a strong portfolio project, prefer one of these:

- AWS S3 or Cloudflare R2 for a production-looking setup.
- GCS if the project is presented as Google Cloud-native.
- Azure Blob if targeting Microsoft/Azure-heavy companies.
- MinIO when demonstrating self-hosted Kubernetes infrastructure.

Do not commit DVC access keys into the repository. Keep them in GitHub Secrets or
Kubernetes Secrets.

Remote-specific DVC dependencies:

| Remote type | Extra package usually needed |
| --- | --- |
| AWS S3 / Cloudflare R2 / MinIO | `dvc-s3` |
| Google Cloud Storage | `dvc-gs` |
| Azure Blob Storage | `dvc-azure` |
| SSH | `dvc-ssh` |
| HTTP/WebDAV | Usually covered by the base DVC install, depending on URL type. |

AWS S3 is now selected, so `dvc-s3` is pinned in `requirements.txt`.

## Portfolio value

This phase proves that the project is not just model code. It has:

- Reproducible data pulls through DVC.
- Automated model training and promotion.
- Container image publishing.
- Manual production deployment with a protected GitHub environment.
- Kubernetes rollout verification.
