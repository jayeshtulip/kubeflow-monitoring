<#
.SYNOPSIS
    Deploy the LLM Platform to staging environment.

.DESCRIPTION
    1. Run L1/L2 tests
    2. Build and push Docker images to ECR
    3. Update ArgoCD image tag
    4. Wait for ArgoCD sync
    5. Run L3 smoke tests

.PARAMETER ImageTag
    Docker image tag to deploy (default: git commit SHA)

.PARAMETER SkipTests
    Skip pre-deploy tests

.EXAMPLE
    .\scripts\deploy\deploy_to_staging.ps1
    .\scripts\deploy\deploy_to_staging.ps1 -ImageTag abc123 -SkipTests
#>
param(
    [string]$ImageTag = (git rev-parse --short HEAD),
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

Write-Host "`n=== LLM Platform Staging Deploy ===" -ForegroundColor Cyan
Write-Host "Image Tag : $ImageTag"
Write-Host "Root      : $Root`n"

# Step 1: Pre-deploy tests
if (-not $SkipTests) {
    Write-Host "Step 1: Running L1/L2 tests..." -ForegroundColor Yellow
    Set-Location $Root
    python -m pytest tests/l1_infrastructure/ tests/l2_component/ -m "l1 or l2" -v --timeout=60
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pre-deploy tests FAILED. Aborting."
        exit 1
    }
    Write-Host "Tests PASSED`n" -ForegroundColor Green
}

# Step 2: Build images
Write-Host "Step 2: Building Docker images..." -ForegroundColor Yellow
$AccountId = aws sts get-caller-identity --query Account --output text
$EcrBase   = "${AccountId}.dkr.ecr.us-east-1.amazonaws.com/llm-platform"

aws ecr get-login-password --region us-east-1 |
    docker login --username AWS --password-stdin "${AccountId}.dkr.ecr.us-east-1.amazonaws.com"

foreach ($img in @("api", "agents")) {
    $tag = "${EcrBase}/${img}:${ImageTag}"
    docker build -t $tag -f "docker/${img}/Dockerfile" $Root
    docker push $tag
    Write-Host "Pushed: $tag" -ForegroundColor Green
}

# Step 3: Update ArgoCD manifest
Write-Host "`nStep 3: Updating ArgoCD manifest..." -ForegroundColor Yellow
(Get-Content "$Root\gitops\apps\staging\llm-platform.yaml") `
    -replace "tag:.*", "tag: $ImageTag" |
    Set-Content "$Root\gitops\apps\staging\llm-platform.yaml"

git -C $Root add gitops/apps/staging/llm-platform.yaml
git -C $Root commit -m "deploy: staging $ImageTag"
git -C $Root push

Write-Host "ArgoCD manifest updated. Sync will start automatically.`n"

# Step 4: Wait for sync
Write-Host "Step 4: Waiting for ArgoCD sync (max 5 min)..." -ForegroundColor Yellow
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    $status = argocd app get llm-platform-api --output json 2>$null |
        ConvertFrom-Json | Select-Object -ExpandProperty status |
        Select-Object -ExpandProperty sync | Select-Object -ExpandProperty status
    if ($status -eq "Synced") {
        Write-Host "ArgoCD Synced!" -ForegroundColor Green
        break
    }
    Write-Host "  Status: $status — waiting 15s..."
    Start-Sleep 15
}

# Step 5: Smoke test
Write-Host "`nStep 5: L3 smoke tests..." -ForegroundColor Yellow
$env:API_BASE_URL = $env:STAGING_API_URL
python -m pytest tests/l3_integration/workflows/test_workflows_e2e.py `
    -k "test_health or test_simple" -v --timeout=60
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDeployment to staging COMPLETE." -ForegroundColor Green
} else {
    Write-Error "Smoke tests FAILED. Check staging environment."
}
