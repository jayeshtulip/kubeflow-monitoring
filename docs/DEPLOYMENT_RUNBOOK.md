# Production Deployment Runbook
**Account:** 659071697671  |  **Region:** us-east-1  |  **Last updated:** March 2026

---

## Prerequisites

```powershell
# Verify AWS CLI works
aws sts get-caller-identity
# Should show Account: 659071697671

# Install required tools
choco install terraform kubectl helm -y   # or use winget
pip install awscli kfp==2.4.0
```

---

## Phase 1 — Terraform (run once quotas approved)

```powershell
cd "C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS\infra\terraform\environments\prod"

# Set your DB password
$env:TF_VAR_db_password = "YourSecurePassword123!"

# Init (uses S3 backend already created)
terraform init

# Preview
terraform plan -out=tfplan

# Apply (~15 min)
terraform apply tfplan
```

After apply, note the outputs:
```powershell
terraform output rds_endpoint   # → fill POSTGRES_HOST in .env.prod
terraform output redis_endpoint # → fill REDIS_HOST in .env.prod
terraform output cluster_name   # → llm-platform-prod
```

---

## Phase 2 — Connect kubectl to EKS

```powershell
aws eks update-kubeconfig --name llm-platform-prod --region us-east-1
kubectl get nodes   # should show 2x t3.xlarge + 1x g4dn.2xlarge
```

---

## Phase 3 — Install cluster components

```powershell
# NVIDIA GPU operator (enables GPU on g4dn nodes)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator --namespace gpu-operator --create-namespace

# Kubeflow Pipelines
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=2.2.0"

# Qdrant
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm install qdrant qdrant/qdrant --namespace llm-platform-prod --create-namespace \
  --set persistence.size=20Gi \
  --set resources.requests.memory=2Gi

# MLflow
helm repo add community-charts https://community-charts.github.io/helm-charts
helm install mlflow community-charts/mlflow --namespace mlflow --create-namespace \
  --set backendStore.postgres.host=$(terraform output -raw rds_endpoint) \
  --set backendStore.postgres.database=llm_platform \
  --set backendStore.postgres.username=llm_admin \
  --set artifactRoot.s3.bucket=llm-platform-mlflow-artifacts-659071697671

# Ollama (Mistral-7B in-cluster)
helm repo add ollama https://otwld.github.io/ollama-helm
helm install ollama ollama/ollama --namespace llm-platform-prod \
  --set ollama.models[0]=mistral:7b \
  --set resources.requests.memory=8Gi
```

---

## Phase 4 — Build and push Docker images

```powershell
cd "C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS"

# Login to ECR
aws ecr get-login-password --region us-east-1 | `
  docker login --username AWS --password-stdin 659071697671.dkr.ecr.us-east-1.amazonaws.com

# Build and push API
$TAG = git rev-parse --short HEAD
docker build -t llm-platform/api:$TAG -f docker/api/Dockerfile .
docker tag llm-platform/api:$TAG 659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/api:$TAG
docker push 659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/api:$TAG

# Build and push agents
docker build -t llm-platform/agents:$TAG -f docker/agents/Dockerfile .
docker tag llm-platform/agents:$TAG 659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/agents:$TAG
docker push 659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/agents:$TAG
```

---

## Phase 5 — Deploy platform

```powershell
# Create namespaces
kubectl apply -f infra/k8s/namespaces.yaml

# Deploy vLLM (Mistral-7B-GPTQ on T4)
kubectl apply -f infra/k8s/vllm/vllm-deployment.yaml

# Create secrets
kubectl create secret generic llm-platform-secrets \
  --namespace llm-platform-prod \
  --from-literal=postgres-password=$env:DB_PASSWORD \
  --from-literal=langsmith-api-key=$env:LANGSMITH_API_KEY \
  --from-literal=hf-token=$env:HF_TOKEN

# Deploy platform via Helm
helm upgrade --install llm-platform infra/helm/llm-platform/ \
  --namespace llm-platform-prod \
  -f infra/helm/llm-platform/values-prod.yaml \
  --set image.registry=659071697671.dkr.ecr.us-east-1.amazonaws.com \
  --set image.api.tag=$TAG \
  --set image.agents.tag=$TAG

# Wait for rollout
kubectl rollout status deployment/llm-platform-api -n llm-platform-prod
kubectl rollout status deployment/llm-platform-agents -n llm-platform-prod
```

---

## Phase 6 — Seed data and verify

```powershell
# Port-forward PostgreSQL seed
$env:POSTGRES_HOST = $(terraform output -raw rds_endpoint).split(":")[0]
$env:POSTGRES_PASSWORD = $env:TF_VAR_db_password

python scripts/data/seed_golden_qa.py

# Port-forward Kubeflow UI
kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow &
Start-Process "http://localhost:8080"

# Run health check
$env:PYTHONPATH = (Get-Location).Path
.\scripts\debug\check_platform_health.ps1
```

---

## Phase 7 — Run L1 + L2 tests against live cluster

```powershell
# Set env vars from terraform outputs
$env:POSTGRES_HOST = $(terraform output -raw rds_endpoint).split(":")[0]
$env:REDIS_HOST    = $(terraform output -raw redis_endpoint)
$env:QDRANT_HOST   = "localhost"  # port-forwarded

# Port-forward Qdrant
kubectl port-forward svc/qdrant-service 6333:6333 -n llm-platform-prod &

python -m pytest tests/l1_infrastructure/ -m l1 -v
python -m pytest tests/l2_component/ -m "l2 and not live" -v
```

---

## Quota summary

| Resource | Requested | Used | Remaining |
|---|---|---|---|
| Standard vCPU | 32 | 8 (2x t3.xlarge) | 24 |
| G-vCPU | 8 | 8 (1x g4dn.2xlarge) | 0 |
| RDS | 1 | 1 | — |
| ElastiCache | 1 | 1 | — |

---

## Cost estimate (us-east-1, on-demand)

| Resource | Type | $/hour | $/month |
|---|---|---|---|
| EKS workers | 2x t3.xlarge | $0.166 | ~$120 |
| GPU node | g4dn.2xlarge | $0.752 | ~$540 |
| RDS | db.t3.medium | $0.068 | ~$49 |
| Redis | cache.t3.micro | $0.017 | ~$12 |
| NAT Gateway | — | $0.045 | ~$32 |
| S3 + data | — | — | ~$5 |
| **Total** | | | **~$758/month** |

> **Cost saving tip:** Scale GPU node to 0 when not actively testing:
> `aws eks update-nodegroup-config --cluster-name llm-platform-prod --nodegroup-name gpu-worker --scaling-config minSize=0,maxSize=1,desiredSize=0`
