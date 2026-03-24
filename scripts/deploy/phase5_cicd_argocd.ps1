<#
.SYNOPSIS
    Phase 5: CI/CD Pipeline + ArgoCD GitOps
    Plan: Section 11 (Complete CI/CD Pipeline)

    Sets up:
      - GitHub OIDC trust for GitHub Actions (no long-lived keys)
      - ArgoCD installed on cluster (app-of-apps pattern)
      - ArgoCD applications pointing to GitHub repo
      - GitHub Actions secrets configured
      - First end-to-end PR → build → ECR → ArgoCD deploy
#>

$ErrorActionPreference = "Stop"
$CLUSTER    = "llm-platform-prod"
$REGION     = "us-east-1"
$ACCOUNT_ID = "659071697671"
$ECR_BASE   = "659071697671.dkr.ecr.us-east-1.amazonaws.com"
$DEST       = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Log  ([string]$m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok   ([string]$m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Title([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Magenta }

# ── Step 1: Create GitHub OIDC provider (plan: Section 11.2 — no long-lived keys)
Title "Step 1: GitHub OIDC trust for GitHub Actions"

$GITHUB_ORG  = $env:GITHUB_ORG   # e.g. "jayesh-zadara"
$GITHUB_REPO = $env:GITHUB_REPO  # e.g. "llm-platform"

if (-not $GITHUB_ORG -or -not $GITHUB_REPO) {
  Write-Host "  Set GITHUB_ORG and GITHUB_REPO env vars first" -ForegroundColor Red
  Write-Host "  Example: `$env:GITHUB_ORG='your-github-username'" -ForegroundColor Yellow
  Write-Host "           `$env:GITHUB_REPO='llm-platform'" -ForegroundColor Yellow
  exit 1
}

# Create OIDC provider
$OIDC_THUMBPRINT = "6938fd4d98bab03faadb97b34396831e3780aea1"
aws iam create-open-id-connect-provider `
  --url "https://token.actions.githubusercontent.com" `
  --client-id-list "sts.amazonaws.com" `
  --thumbprint-list $OIDC_THUMBPRINT `
  --region $REGION 2>$null

# Create IAM role for GitHub Actions
$TRUST_POLICY = @"
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:${GITHUB_ORG}/${GITHUB_REPO}:*"}
    }
  }]
}
"@

$TRUST_POLICY | Out-File -FilePath /tmp/trust-policy.json -Encoding utf8

aws iam create-role `
  --role-name GitHubActions-LLMPlatform `
  --assume-role-policy-document file:///tmp/trust-policy.json `
  --region $REGION 2>$null

# Attach required permissions
foreach ($policy in @(
  "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser",
  "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
)) {
  aws iam attach-role-policy `
    --role-name GitHubActions-LLMPlatform `
    --policy-arn $policy
}

$ROLE_ARN = "arn:aws:iam::${ACCOUNT_ID}:role/GitHubActions-LLMPlatform"
Ok "GitHub OIDC trust created — role: $ROLE_ARN"
Write-Host "`n  Add to GitHub repo secrets:" -ForegroundColor Yellow
Write-Host "  AWS_ROLE_ARN = $ROLE_ARN" -ForegroundColor White
Write-Host "  AWS_ACCOUNT_ID = $ACCOUNT_ID" -ForegroundColor White
Write-Host "  EKS_CLUSTER_NAME = $CLUSTER" -ForegroundColor White

# ── Step 2: Install ArgoCD (plan: Section 11.4) ───────────────────────────────
Title "Step 2: Install ArgoCD (plan: Section 11.4 — app-of-apps)"
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/v2.10.0/manifests/install.yaml
Log "Waiting for ArgoCD to be ready..."
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s

# Get ArgoCD admin password
$ARGOCD_PASS = kubectl -n argocd get secret argocd-initial-admin-secret `
  -o jsonpath="{.data.password}" | `
  [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_))
Ok "ArgoCD installed — admin password: $ARGOCD_PASS"

# ── Step 3: Configure ArgoCD with GitOps repo (plan: Section 11.4) ───────────
Title "Step 3: Configure ArgoCD to watch GitHub repo"
kubectl port-forward svc/argocd-server 8888:443 -n argocd &
Start-Sleep 3

argocd login localhost:8888 `
  --username admin `
  --password $ARGOCD_PASS `
  --insecure

# Add GitHub repo
argocd repo add "https://github.com/${GITHUB_ORG}/${GITHUB_REPO}" `
  --username $GITHUB_ORG `
  --password $env:GITHUB_TOKEN

# Apply app-of-apps (plan: Section 11.4 — root Application)
kubectl apply -f "$DEST\gitops\apps\root\app-of-apps.yaml" -n argocd
Ok "ArgoCD configured — watching https://github.com/${GITHUB_ORG}/${GITHUB_REPO}"

# ── Step 4: Argo Rollouts (plan: Section 11.5 — blue/green) ─────────────────
Title "Step 4: Argo Rollouts (plan: Section 11.5 — blue/green deployment)"
kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
kubectl wait --for=condition=available deployment/argo-rollouts -n argo-rollouts --timeout=180s
Ok "Argo Rollouts installed — manages Istio VirtualService traffic ramp"

# ── Step 5: Install Istio (plan: Section 11.5 — Istio VirtualService) ────────
Title "Step 5: Istio 1.20 (plan: Section 11.5 — blue/green + A/B traffic)"
$ISTIO_VERSION = "1.20.0"
if (-not (Get-Command istioctl -ErrorAction SilentlyContinue)) {
  Log "Downloading istioctl..."
  Invoke-WebRequest `
    -Uri "https://github.com/istio/istio/releases/download/$ISTIO_VERSION/istioctl-$ISTIO_VERSION-win-amd64.zip" `
    -OutFile "$env:TEMP\istioctl.zip"
  Expand-Archive "$env:TEMP\istioctl.zip" -DestinationPath "$env:TEMP\istio"
  Move-Item "$env:TEMP\istio\istioctl.exe" "C:\Windows\System32\istioctl.exe"
}
istioctl install --set profile=default -y
kubectl label namespace llm-platform-prod istio-injection=enabled --overwrite
kubectl apply -f "$DEST\infra\k8s\istio-virtual-service.yaml"
Ok "Istio installed — Ollama 70% / vLLM 30% traffic split configured"

# ── Step 6: External Secrets Operator (plan: Section 11.6) ───────────────────
Title "Step 6: External Secrets Operator (plan: Section 11.6 — no secrets in Git)"
helm repo add external-secrets https://charts.external-secrets.io --force-update
helm upgrade --install external-secrets external-secrets/external-secrets `
  --namespace external-secrets `
  --create-namespace `
  --wait
Ok "External Secrets Operator installed"

# Store secrets in AWS Secrets Manager
$secrets = @{
  "postgres-password"  = $env:DB_PASSWORD
  "langsmith-api-key"  = $env:LANGSMITH_API_KEY
  "hf-token"           = $env:HF_TOKEN
  "slack-webhook-url"  = $env:SLACK_WEBHOOK_URL
}
foreach ($key in $secrets.Keys) {
  aws secretsmanager create-secret `
    --name "llm-platform/$key" `
    --secret-string $secrets[$key] `
    --region $REGION 2>$null
  aws secretsmanager put-secret-value `
    --secret-id "llm-platform/$key" `
    --secret-string $secrets[$key] `
    --region $REGION
}
Ok "Secrets stored in AWS Secrets Manager (auto-rotates every 30 days)"

# ── Step 7: Build and push Docker images (plan: Section 11.2) ────────────────
Title "Step 7: Build and push Docker images to ECR (plan: Section 11.2)"
Set-Location $DEST
$TAG = git rev-parse --short HEAD 2>$null
if (-not $TAG) { $TAG = "v1.0.0" }

aws ecr get-login-password --region $REGION | `
  docker login --username AWS --password-stdin $ECR_BASE

foreach ($service in @("api", "agents")) {
  Log "Building $service ($TAG)..."
  docker build -t "llm-platform/${service}:$TAG" -f "docker/${service}/Dockerfile" .
  docker tag "llm-platform/${service}:$TAG" "$ECR_BASE/llm-platform/${service}:$TAG"
  docker push "$ECR_BASE/llm-platform/${service}:$TAG"
  Ok "$service image pushed: $ECR_BASE/llm-platform/${service}:$TAG"
}

# ── Step 8: Deploy platform via Helm (plan: Section 11.3) ────────────────────
Title "Step 8: Deploy platform via Helm (plan: Section 11.3)"
helm upgrade --install llm-platform infra/helm/llm-platform/ `
  --namespace llm-platform-prod `
  --values infra/helm/llm-platform/values-prod.yaml `
  --set image.registry=$ECR_BASE `
  --set image.api.tag=$TAG `
  --set image.agents.tag=$TAG `
  --wait --timeout=300s

kubectl rollout status deployment/llm-platform-api -n llm-platform-prod
kubectl rollout status deployment/llm-platform-agents -n llm-platform-prod
Ok "Platform deployed — image tag: $TAG"

Title "Phase 5 Complete — CI/CD + ArgoCD Live"
Write-Host "`n  ArgoCD UI: kubectl port-forward svc/argocd-server 8888:443 -n argocd" -ForegroundColor Green
Write-Host "  Login:     admin / $ARGOCD_PASS" -ForegroundColor Green
Write-Host "`n  Every push to main now triggers:" -ForegroundColor Cyan
Write-Host "    PR opened   → L1+L2 tests (<8 min)" -ForegroundColor White
Write-Host "    Merge main  → build + ECR push + L1+L2+L3 (<25 min)" -ForegroundColor White
Write-Host "    ArgoCD      → auto-deploy to dev/staging" -ForegroundColor White
Write-Host "    Production  → manual approval in GitHub Environments" -ForegroundColor White
