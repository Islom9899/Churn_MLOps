# AWS EC2 smoke test

MLOps step: Phase 5, low-cost cloud deployment smoke test before Kubernetes.

This test proves that a fresh AWS machine can pull DVC artifacts from S3, build
the API container, load the serving model, answer `/health`, answer `/predict`,
and expose `/metrics`.

## AWS console settings

Use a short-lived EC2 instance. Terminate it after the test.

Recommended settings for the first test:

| Field | Value |
| --- | --- |
| Region | `ap-northeast-2` |
| AMI | Amazon Linux 2023 |
| Architecture | `x86_64` |
| Instance type | `t3.small` |
| Storage | 20 GiB gp3 |
| Inbound SSH | TCP 22 from My IP |
| Inbound API | TCP 8000 from My IP |
| Database | `sqlite:////tmp/churn-api-smoke.db` for this smoke test |

Do not create a NAT Gateway or Load Balancer for this test.

## EC2 commands

Run these commands after connecting with SSH.

```bash
sudo dnf update -y
# Updates Amazon Linux packages before installing tools.

sudo dnf install -y git docker python3 python3-pip curl
# Installs Git, Docker, Python, pip, and curl.

sudo systemctl enable --now docker
# Starts Docker now and enables it after reboot.

git clone https://github.com/Islom9899/Churn_MLOps.git
# Downloads the project source code from GitHub.

cd Churn_MLOps
# Enters the project root directory.

bash scripts/aws_ec2_smoke_test.sh
# Pulls the DVC model from S3, builds Docker, runs the API, and checks endpoints.
```

## Expected result

The script should print:

```text
{"status":"ok"}
```

Then `/predict` should return `churn`, `churn_probability`, and `churn_label`.

Then `/metrics` should contain:

```text
churn_api_http_requests_total
churn_api_predictions_total
```

## Stop costs

When the test is done:

```bash
sudo docker rm -f churn-api-smoke
# Stops and deletes the running API container.
```

Then terminate the EC2 instance in the AWS console.
