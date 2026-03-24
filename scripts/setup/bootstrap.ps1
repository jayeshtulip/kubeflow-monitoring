<#
.SYNOPSIS
    Bootstrap the LLM Platform Windows development environment.

.DESCRIPTION
    Installs all required tools and Python packages.
    Run this once after cloning the repository.

.EXAMPLE
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\scripts\setup\bootstrap.ps1
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Log($msg) { Write-Host $msg -ForegroundColor Cyan }
function OK($msg)  { Write-Host "OK   $msg" -ForegroundColor Green }
function WARN($msg){ Write-Host "WARN $msg" -ForegroundColor Yellow }

Log "`n=== LLM Platform Bootstrap ===" 
Log "Project root: $Root`n"

Set-Location $Root

# ── 1. Python version check ────────────────────────────────────────────────
Log "Step 1: Python version check"
$pyver = python --version 2>&1
if ($pyver -notmatch "3\.1[01]") {
    Write-Error "Python 3.10 or 3.11 required. Found: $pyver"
    exit 1
}
OK "Python: $pyver"

# ── 2. Upgrade pip ────────────────────────────────────────────────────────
Log "`nStep 2: Upgrade pip"
python -m pip install --upgrade pip
OK "pip upgraded"

# ── 3. Install core Python packages ───────────────────────────────────────
Log "`nStep 3: Install Python packages"
python -m pip install `
    mlflow qdrant-client sentence-transformers `
    "kfp==2.4.0" "kfp-kubernetes==1.0.0" `
    langchain "langgraph==0.1.1" langsmith `
    psycopg2-binary redis boto3 `
    fastapi uvicorn httpx `
    ragas datasets `
    prometheus-client `
    pydantic sqlalchemy `
    pytest pytest-asyncio pytest-timeout `
    dvc ruff
OK "Python packages installed"

# ── 4. Create __init__.py files ───────────────────────────────────────────
Log "`nStep 4: Ensure __init__.py files"
$pkgDirs = @(
    "src", "src\storage", "src\storage\qdrant", "src\storage\redis",
    "src\storage\postgres", "src\agents", "src\agents\router",
    "src\agents\planner", "src\agents\executor", "src\agents\critic",
    "src\agents\tools", "src\workflows", "src\workflows\simple_research",
    "src\workflows\react", "src\workflows\smart_tools",
    "src\workflows\planner_executor_critic",
    "src\serving", "src\serving\ollama", "src\serving\vllm",
    "src\guardrails", "src\api", "src\api\routers", "src\api\schemas",
    "src\drift", "src\drift\evidently",
    "src\observability", "src\observability\langsmith",
    "src\observability\prometheus", "src\observability\grafana_exporter",
    "pipelines", "pipelines\components", "pipelines\components\shared",
    "mlops", "mlops\model_registry", "mlops\lineage"
)
foreach ($d in $pkgDirs) {
    $init = Join-Path $Root "$d\__init__.py"
    if (-not (Test-Path $init)) {
        New-Item $init -ItemType File -Force | Out-Null
        Write-Host "  Created $init"
    }
}
OK "__init__.py files ensured"

# ── 5. Copy .env.example -> .env ──────────────────────────────────────────
Log "`nStep 5: Environment file"
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $Root ".env.example") $envFile
    OK ".env created from .env.example — fill in your values"
} else {
    WARN ".env already exists — skipping"
}

# ── 6. Run import checks ──────────────────────────────────────────────────
Log "`nStep 6: Import checks"
$env:PYTHONPATH = $Root

$checks = @(
    "from pipelines.components.shared.base import EnvConfig; print('OK base')",
    "from src.agents.router.workflow_router import route_query; print('OK router')",
    "from src.guardrails.input_validator import validate_input; print('OK guardrails')",
    "from src.api.main import app; print('OK FastAPI')",
    "from pipelines.p05_data_indexing.pipeline import data_indexing_pipeline; print('OK P05')",
    "from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline; print('OK P09')",
    "from pipelines.p11_automated_retraining.pipeline import automated_retraining_pipeline; print('OK P11')"
)

$passed = 0
foreach ($check in $checks) {
    try {
        python -c $check 2>&1
        $passed++
    } catch {
        Write-Host "FAIL: $check" -ForegroundColor Red
    }
}
OK "$passed/$($checks.Count) import checks passed"

# ── 7. Summary ────────────────────────────────────────────────────────────
Write-Host "`n=== Bootstrap Complete ===" -ForegroundColor Green
Write-Host @"

Next steps:
  1. Edit .env and fill in your PostgreSQL, Redis, and AWS credentials
  2. Seed golden QA data:
       python scripts/data/seed_golden_qa.py
  3. Run import checks:
       python -c "from src.api.main import app; print('OK')"
  4. When EKS cluster is ready, compile pipelines:
       python pipelines/p05_data_indexing/pipeline.py
       python pipelines/p09_ragas_evaluation/pipeline.py
       python pipelines/p11_automated_retraining/pipeline.py
  5. Deploy to staging:
       .\scripts\deploy\deploy_to_staging.ps1

"@
