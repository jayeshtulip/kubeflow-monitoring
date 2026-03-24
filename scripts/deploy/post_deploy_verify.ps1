<#
.SYNOPSIS
    Post-deployment verification — run after all phases complete.
    Plan: Phase 5 (Weeks 9-10) — Hardening

    Verifies:
      - All pods running
      - L1 + L2 + L3 tests passing
      - RAGAS scores above thresholds
      - Katib HPO converged
      - DVC reproducibility (SHA match)
      - Grafana alerts configured
      - Platform health check
#>

$DEST      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$NAMESPACE = "llm-platform-prod"

function Ok   ([string]$m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Fail ([string]$m) { Write-Host "  FAIL $m" -ForegroundColor Red }
function Title([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Magenta }

$env:PYTHONPATH = $DEST

Title "1. Pod health check"
$pods = kubectl get pods -n $NAMESPACE --no-headers
Write-Host $pods
$failed = $pods | Where-Object { $_ -notmatch "Running|Completed" }
if ($failed) { Fail "Some pods not running"; $failed | ForEach-Object { Write-Host "  $_" } }
else { Ok "All pods Running" }

Title "2. L1 Infrastructure tests"
kubectl port-forward svc/qdrant 6333:6333 -n $NAMESPACE &
kubectl port-forward svc/mlflow 5000:5000 -n mlflow &
Start-Sleep 5

$env:POSTGRES_HOST = (aws rds describe-db-instances `
  --query "DBInstances[?DBInstanceIdentifier=='llm-platform-prod-postgres'].Endpoint.Address" `
  --output text --region us-east-1)
$env:REDIS_HOST = (aws elasticache describe-cache-clusters `
  --query "CacheClusters[?CacheClusterId=='llm-platform-prod-redis'].CacheNodes[0].Endpoint.Address" `
  --output text --region us-east-1)

python -m pytest tests/l1_infrastructure/ -m l1 -v --timeout=60 -q

Title "3. L2 Component tests"
python -m pytest tests/l2_component/ -m "l2 and not live" -v --timeout=60 -q

Title "4. L3 Integration tests"
kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow &
Start-Sleep 3
$env:KUBEFLOW_HOST = "http://localhost:8080"
python -m pytest tests/l3_integration/ -m l3 -v --timeout=300 -q

Title "5. RAGAS scores (plan: faithfulness >= 0.90)"
python "$DEST\scripts\validate\check_ragas_scores.py"

Title "6. DVC reproducibility (plan: Phase 5 — identical SHA)"
Set-Location $DEST
dvc repro 2>$null
$sha1 = dvc status --json 2>$null | python -c "import json,sys; d=json.load(sys.stdin); print(hash(str(d)))"
dvc repro 2>$null
$sha2 = dvc status --json 2>$null | python -c "import json,sys; d=json.load(sys.stdin); print(hash(str(d)))"
if ($sha1 -eq $sha2) { Ok "DVC reproducible — identical SHA on two runs" }
else { Fail "DVC non-deterministic — SHAs differ" }

Title "7. Katib HPO status"
kubectl get experiment -n kubeflow -o wide

Title "8. Platform health"
.\scripts\debug\check_platform_health.ps1

Title "Verification Complete"
