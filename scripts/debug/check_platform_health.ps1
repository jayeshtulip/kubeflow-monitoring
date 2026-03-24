<#
.SYNOPSIS
    Quick health check of all LLM Platform components.
    Run this when you suspect something is wrong.
#>
param(
    [string]$ApiUrl    = $env:API_BASE_URL,
    [string]$Namespace = "llm-platform-prod"
)

$ErrorActionPreference = "Continue"

function Check($name, $cmd) {
    try {
        $result = & $cmd
        Write-Host "OK   $name" -ForegroundColor Green
        return $result
    } catch {
        Write-Host "FAIL $name : $_" -ForegroundColor Red
        return $null
    }
}

Write-Host "`n=== LLM Platform Health Check ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')`n"

# Kubernetes pods
Write-Host "--- Kubernetes ---" -ForegroundColor Yellow
kubectl get pods -n $Namespace 2>&1 | Where-Object { $_ -notmatch "^NAME" } |
    ForEach-Object {
        $col = if ($_ -match "Running") { "Green" } else { "Red" }
        Write-Host "  $_" -ForegroundColor $col
    }

# API health
Write-Host "`n--- API ---" -ForegroundColor Yellow
if ($ApiUrl) {
    $r = Invoke-RestMethod "$ApiUrl/health" -ErrorAction SilentlyContinue
    if ($r.status -eq "ok") {
        Write-Host "OK   /health" -ForegroundColor Green
    } else {
        Write-Host "WARN /health: $($r | ConvertTo-Json)" -ForegroundColor Yellow
    }
    $r2 = Invoke-RestMethod "$ApiUrl/ready" -ErrorAction SilentlyContinue
    if ($r2) {
        Write-Host "  ollama_available  : $($r2.ollama_available)"
        Write-Host "  vllm_available    : $($r2.vllm_available)"
        Write-Host "  qdrant_available  : $($r2.qdrant_available)"
    }
} else {
    Write-Host "SKIP API_BASE_URL not set" -ForegroundColor Yellow
}

# Qdrant
Write-Host "`n--- Qdrant ---" -ForegroundColor Yellow
$qdrantPort = kubectl get svc qdrant-service -n $Namespace -o jsonpath="{.spec.ports[0].nodePort}" 2>$null
Write-Host "  Qdrant service port: $qdrantPort"

# Redis
Write-Host "`n--- Redis ---" -ForegroundColor Yellow
kubectl exec -n $Namespace `
    (kubectl get pod -n $Namespace -l app=redis -o jsonpath="{.items[0].metadata.name}" 2>$null) `
    -- redis-cli ping 2>$null | ForEach-Object { Write-Host "  redis-cli ping: $_" }

# Recent platform errors
Write-Host "`n--- Recent errors (last 50 lines) ---" -ForegroundColor Yellow
kubectl logs -n $Namespace -l app=llm-platform-api --tail=50 2>$null |
    Where-Object { $_ -match "ERROR|CRITICAL|Exception" } |
    Select-Object -Last 10 |
    ForEach-Object { Write-Host "  $_" -ForegroundColor Red }

Write-Host "`n=== Done ===`n" -ForegroundColor Cyan
