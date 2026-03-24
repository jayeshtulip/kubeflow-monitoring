# Phase 2: Kubeflow Full Edition + DVC + Monitoring
# Plan: Section 2.4 + Phase 2 (Weeks 3-4)
# Account: 659071697671  Cluster: llm-platform-prod

$ErrorActionPreference = "Stop"
$CLUSTER    = "llm-platform-prod"
$REGION     = "us-east-1"
$ACCOUNT_ID = "659071697671"
$NAMESPACE  = "llm-platform-prod"
$DEST       = "C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS"
$RDS_HOST   = "llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com"
$REDIS_HOST = "llm-platform-prod-redis.dho8vz.0001.use1.cache.amazonaws.com"
$DB_PASS    = $env:DB_PASSWORD
if (-not $DB_PASS) { $DB_PASS = "LLMPlatform2026!" }

function Log   ([string]$m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok    ([string]$m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Warn  ([string]$m) { Write-Host "  WARN $m" -ForegroundColor Yellow }
function Title ([string]$m) { Write-Host "" ; Write-Host "=== $m ===" -ForegroundColor Magenta }

# Step 0: Verify cluster
Title "Step 0: Verify cluster connection"
kubectl get nodes
Ok "Cluster connected"

# Step 1: Namespaces
Title "Step 1: Create namespaces"
kubectl apply -f "$DEST\infra\k8s\namespaces.yaml"
Ok "Namespaces ready"

# Step 2: NVIDIA GPU Operator (prepares for when GPU quota approved)
Title "Step 2: NVIDIA GPU Operator"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm upgrade --install gpu-operator nvidia/gpu-operator `
  --namespace gpu-operator --create-namespace `
  --set driver.enabled=true `
  --set toolkit.enabled=true `
  --wait --timeout=300s
Ok "GPU operator installed"

# Step 3: Kubeflow Pipelines v2
Title "Step 3: Kubeflow Pipelines v2"
Log "Applying cluster-scoped resources..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
Start-Sleep 10
Log "Applying platform-agnostic..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.2.0"
Log "Waiting for Kubeflow (~5 min)..."
kubectl wait --for=condition=available deployment/ml-pipeline -n kubeflow --timeout=360s
Ok "Kubeflow Pipelines v2 installed"

# Step 4: Katib HPO
Title "Step 4: Katib HPO (Section 2.4)"
kubectl apply -k "github.com/kubeflow/katib/manifests/v1beta1/installs/katib-standalone?ref=v0.16.0"
kubectl wait --for=condition=available deployment/katib-controller -n kubeflow --timeout=180s
kubectl apply -f "$DEST\models\katib\experiments\rag_chunking_experiment.yaml"
Ok "Katib installed + RAG chunking experiment submitted"

# Step 5: MLflow (PostgreSQL backend + S3 artifacts)
Title "Step 5: MLflow"
Log "RDS endpoint: $RDS_HOST"
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update
helm upgrade --install mlflow community-charts/mlflow `
  --namespace mlflow --create-namespace `
  --set image.tag="v2.12.0" `
  --set "backendStore.postgres.enabled=true" `
  --set "backendStore.postgres.host=$RDS_HOST" `
  --set "backendStore.postgres.port=5432" `
  --set "backendStore.postgres.database=llm_platform" `
  --set "backendStore.postgres.username=llm_admin" `
  --set "backendStore.postgres.password=$DB_PASS" `
  --set "artifactRoot.s3.enabled=true" `
  --set "artifactRoot.s3.bucket=llm-platform-mlflow-artifacts-659071697671" `
  --set "artifactRoot.s3.awsRegion=us-east-1" `
  --wait --timeout=180s
Ok "MLflow installed - RDS backend + S3 artifacts"

# Step 6: Qdrant
Title "Step 6: Qdrant vector store"
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update
helm upgrade --install qdrant qdrant/qdrant `
  --namespace $NAMESPACE `
  --set "persistence.enabled=true" `
  --set "persistence.size=20Gi" `
  --set "resources.requests.memory=2Gi" `
  --set "resources.requests.cpu=500m" `
  --set "resources.limits.memory=8Gi" `
  --set "resources.limits.cpu=2" `
  --wait --timeout=180s
Ok "Qdrant installed with 20Gi persistence"

# Step 7: Ollama / Mistral-7B
Title "Step 7: Ollama Mistral-7B (in-cluster LLM)"
helm repo add ollama https://otwld.github.io/ollama-helm
helm repo update
helm upgrade --install ollama ollama/ollama `
  --namespace $NAMESPACE `
  --set "ollama.models[0]=mistral:7b" `
  --set "resources.requests.memory=8Gi" `
  --set "resources.requests.cpu=2" `
  --set "resources.limits.memory=12Gi" `
  --set "resources.limits.cpu=4" `
  --timeout=600s
Ok "Ollama installed - pulling mistral:7b (takes 5-10 min)"

# Step 8: Prometheus + Grafana
Title "Step 8: Prometheus + Grafana + exporters"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install kube-prometheus prometheus-community/kube-prometheus-stack `
  --namespace monitoring --create-namespace `
  --set "grafana.adminPassword=LLMPlatform2026!" `
  --set "grafana.persistence.enabled=true" `
  --set "grafana.persistence.size=5Gi" `
  --set "prometheus.prometheusSpec.retention=30d" `
  --wait --timeout=300s
Ok "Prometheus + Grafana installed"

# Redis exporter
helm upgrade --install redis-exporter prometheus-community/prometheus-redis-exporter `
  --namespace monitoring `
  --set "redisAddress=redis://${REDIS_HOST}:6379"
Ok "Redis exporter installed"

# PostgreSQL exporter
helm upgrade --install postgres-exporter prometheus-community/prometheus-postgres-exporter `
  --namespace monitoring `
  --set "config.datasource.host=$RDS_HOST" `
  --set "config.datasource.user=llm_admin" `
  --set "config.datasource.password=$DB_PASS" `
  --set "config.datasource.database=llm_platform"
Ok "PostgreSQL exporter installed"

# Step 9: Prometheus alert rules
Title "Step 9: Alert rules (Section 12.2)"
kubectl apply -f "$DEST\monitoring\prometheus\rules\platform_alerts.yaml" -n monitoring
kubectl apply -f "$DEST\monitoring\prometheus\rules\sla_alerts.yaml" -n monitoring
Ok "Alert rules applied"

# Step 10: Grafana dashboards
Title "Step 10: Grafana dashboards (Section 5)"
kubectl create configmap grafana-dashboard-data-health `
  --from-file="$DEST\monitoring\grafana\dashboards\01_data_health.json" `
  -n monitoring --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboard-mlflow `
  --from-file="$DEST\monitoring\grafana\dashboards\02_mlflow_metrics.json" `
  -n monitoring --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboard-performance `
  --from-file="$DEST\monitoring\grafana\dashboards\03_platform_performance.json" `
  -n monitoring --dry-run=client -o yaml | kubectl apply -f -
Ok "Grafana dashboards provisioned"

# Step 11: DVC remote
Title "Step 11: DVC remote setup (Section 4.1)"
Set-Location $DEST
if (-not (Test-Path ".dvc")) { dvc init }
dvc remote add -d myremote "s3://llm-platform-dvc-remote-659071697671" 2>$null
dvc remote modify myremote region us-east-1
Ok "DVC remote: s3://llm-platform-dvc-remote-659071697671"

# Step 12: Seed production PostgreSQL
Title "Step 12: Seed production data"
$env:POSTGRES_HOST     = $RDS_HOST
$env:POSTGRES_DB       = "llm_platform"
$env:POSTGRES_USER     = "llm_admin"
$env:POSTGRES_PASSWORD = $DB_PASS
$env:PYTHONPATH        = $DEST
python "$DEST\scripts\data\seed_golden_qa.py"
Ok "Golden QA pairs seeded to production PostgreSQL"

# Done
Title "Phase 2 Complete!"
Write-Host ""
Write-Host "  Access the UIs (run each in a separate terminal):" -ForegroundColor Yellow
Write-Host "  Kubeflow:  kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow" -ForegroundColor White
Write-Host "  Katib:     kubectl port-forward svc/katib-ui 8081:80 -n kubeflow" -ForegroundColor White
Write-Host "  MLflow:    kubectl port-forward svc/mlflow 5000:5000 -n mlflow" -ForegroundColor White
Write-Host "  Grafana:   kubectl port-forward svc/kube-prometheus-grafana 3000:80 -n monitoring" -ForegroundColor White
Write-Host "  Qdrant:    kubectl port-forward svc/qdrant 6333:6333 -n llm-platform-prod" -ForegroundColor White
Write-Host ""
Write-Host "  Grafana login: admin / LLMPlatform2026!" -ForegroundColor Cyan
Write-Host "  Next: wait for Ollama to finish pulling mistral:7b then run phase3" -ForegroundColor Cyan
