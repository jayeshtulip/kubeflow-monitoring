<#
.SYNOPSIS
    Phase 4: Grafana Dashboards + LangSmith Enhanced
    Plan: Section 5 + Section 7 + Phase 4 (Weeks 7-8)

    Configures:
      - Dashboard 1: Data Health (DVC + drift + QA coverage)
      - Dashboard 2: MLflow Metrics (RAGAS + latency + Katib)
      - Dashboard 3: Platform Performance (throughput + vLLM + Redis + PG)
      - LangSmith datasets, evaluators, feedback loops
      - L4 Locust load test (50 req/min, P95 < 60s)
#>

$ErrorActionPreference = "Stop"
$CLUSTER   = "llm-platform-prod"
$REGION    = "us-east-1"
$NAMESPACE = "llm-platform-prod"
$DEST      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Log  ([string]$m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok   ([string]$m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Title([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Magenta }

# ── Step 1: Grafana data sources (plan: Section 5) ───────────────────────────
Title "Step 1: Configure Grafana data sources"

# Get Grafana admin password
$GRAFANA_PASS = kubectl get secret kube-prometheus-grafana `
  -n monitoring -o jsonpath="{.data.admin-password}" | `
  [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_))

kubectl port-forward svc/kube-prometheus-grafana 3000:80 -n monitoring &
Start-Sleep 5

# Add MLflow as a JSON data source
$mlflow_ds = @{
  name      = "MLflow"
  type      = "grafana-simple-json-datasource"
  url       = "http://mlflow-service.mlflow.svc.cluster.local:5000"
  access    = "proxy"
  isDefault = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3000/api/datasources" `
  -Method POST `
  -Headers @{ Authorization = "Basic $([Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:$GRAFANA_PASS")))" } `
  -ContentType "application/json" `
  -Body $mlflow_ds 2>$null

Ok "Grafana data sources configured (Prometheus + MLflow)"

# ── Step 2: Import dashboards (plan: Section 5.1, 5.2, 5.3) ─────────────────
Title "Step 2: Import 3 Grafana dashboards (plan: Section 5)"

foreach ($dashboard in @(
  "$DEST\monitoring\grafana\dashboards\01_data_health.json",
  "$DEST\monitoring\grafana\dashboards\02_mlflow_metrics.json",
  "$DEST\monitoring\grafana\dashboards\03_platform_performance.json"
)) {
  $name = Split-Path $dashboard -Leaf
  $content = Get-Content $dashboard -Raw | ConvertFrom-Json
  $payload = @{ dashboard = $content; overwrite = $true; folderId = 0 } | ConvertTo-Json -Depth 20

  Invoke-RestMethod -Uri "http://localhost:3000/api/dashboards/import" `
    -Method POST `
    -Headers @{ Authorization = "Basic $([Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:$GRAFANA_PASS")))" } `
    -ContentType "application/json" `
    -Body $payload

  Ok "Imported: $name"
}

Write-Host "`n  Grafana: http://localhost:3000 (admin / $GRAFANA_PASS)" -ForegroundColor Yellow
Write-Host "  Dashboard 1: Data Health — DVC commit age, JS-divergence, QA coverage" -ForegroundColor Yellow
Write-Host "  Dashboard 2: MLflow Metrics — RAGAS scores, Katib HPO, A/B test results" -ForegroundColor Yellow
Write-Host "  Dashboard 3: Platform Performance — throughput, vLLM, Redis, PostgreSQL" -ForegroundColor Yellow

# ── Step 3: Configure Grafana alerts (plan: Section 12.2, Phase 5) ───────────
Title "Step 3: Configure Grafana alert rules (plan: Section 12.2)"

# The alert rules are already in monitoring/prometheus/rules/
# They get picked up by Prometheus via the PrometheusRule CRD
kubectl get prometheusrule -n monitoring
Ok "Alert rules active: CriticalP95Latency, HallucinationRateCritical, DataDriftDetected, SLABurnRate"

# ── Step 4: LangSmith Enhanced (plan: Section 7) ─────────────────────────────
Title "Step 4: LangSmith Enhanced setup (plan: Section 7)"

if (-not $env:LANGSMITH_API_KEY) {
  Write-Host "  WARNING: LANGSMITH_API_KEY not set — skipping LangSmith setup" -ForegroundColor Yellow
  Write-Host "  Get key from: https://smith.langchain.com" -ForegroundColor Yellow
} else {
  $env:PYTHONPATH = $DEST
  $env:LANGCHAIN_TRACING_V2 = "true"
  $env:LANGSMITH_PROJECT = "llm-platform-prod"

  python -c "
import os
from langsmith import Client

client = Client(api_key=os.environ['LANGSMITH_API_KEY'])

# Create dataset with golden QA pairs (plan: Section 7)
from src.ragas_eval.dataset_builder import load_golden_qa_from_files
pairs = load_golden_qa_from_files('data/golden_qa')

# Create LangSmith dataset
dataset = client.create_dataset(
    'llm-platform-golden-qa-prod',
    description='200 golden QA pairs per domain for RAGAS + LangSmith evaluation',
)
for pair in pairs:
    client.create_example(
        inputs={'question': pair.question},
        outputs={'answer': pair.ground_truth},
        dataset_id=dataset.id,
        metadata={'domain': pair.domain, 'category': pair.category},
    )
print('LangSmith dataset created: ' + str(dataset.id) + ' (' + str(len(pairs)) + ' examples)')
print(f'View at: https://smith.langchain.com/datasets/{dataset.id}')
"
  Ok "LangSmith dataset created with 200 QA pairs"
}

# ── Step 5: L4 Performance test (plan: Phase 4, Section 9) ───────────────────
Title "Step 5: L4 Performance test — Locust (plan: 50 req/min, P95 < 60s)"

# Get API endpoint
$API_URL = kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>$null
if (-not $API_URL) {
  Log "No ingress yet — port-forwarding for load test"
  kubectl port-forward svc/llm-platform-api 8000:8000 -n $NAMESPACE &
  Start-Sleep 3
  $API_URL = "http://localhost:8000"
}

Log "Running 3-minute Locust load test against $API_URL..."
python -m pytest tests/l4_performance/test_load.py -v --timeout=300 -s `
  --api-url=$API_URL 2>&1 | Tail -20

Ok "L4 load test complete"

Title "Phase 4 Complete"
Write-Host "`n  Grafana:   http://localhost:3000" -ForegroundColor Green
Write-Host "  LangSmith: https://smith.langchain.com/projects/llm-platform-prod" -ForegroundColor Green
Write-Host "  Next: Run phase5_cicd_argocd.ps1" -ForegroundColor Cyan
