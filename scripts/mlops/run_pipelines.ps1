<#
.SYNOPSIS
    Submit P05 (Data Indexing) and/or P09 (RAGAS Evaluation) Kubeflow pipelines.

.PARAMETER Pipeline
    Which pipeline to run: p05, p09, or both (default: both)

.PARAMETER DocId
    Document ID for P05 (e.g. JIRA-1234)

.PARAMETER DocFile
    Path to a .txt file to index via P05

.PARAMETER QaLimit
    Number of QA pairs to evaluate in P09 (default: 200)

.PARAMETER Domain
    Domain filter for P09: tech, hr, org, or empty for all

.PARAMETER KubeflowHost
    Kubeflow Pipelines URL (defaults to $env:KUBEFLOW_HOST or port-forward)

.EXAMPLE
    .\scripts\mlops\run_pipelines.ps1 -Pipeline p09 -QaLimit 50
    .\scripts\mlops\run_pipelines.ps1 -Pipeline p05 -DocFile docs\incident.txt -DocId JIRA-1234
    .\scripts\mlops\run_pipelines.ps1 -Pipeline both -QaLimit 200
#>
param(
    [ValidateSet("p05", "p09", "both")]
    [string]$Pipeline     = "both",
    [string]$DocId        = "",
    [string]$DocFile      = "",
    [string]$Source       = "manual",
    [string]$Collection   = "tech_docs",
    [int]   $QaLimit      = 200,
    [string]$Domain       = "",
    [string]$KubeflowHost = $env:KUBEFLOW_HOST,
    [string]$Env          = "staging"
)

$ErrorActionPreference = "Stop"
$Root = (Get-Location).Path
$env:PYTHONPATH = $Root

function Info  ([string]$m) { Write-Host "  [INFO] $m" -ForegroundColor Cyan }
function Ok    ([string]$m) { Write-Host "  [OK]   $m" -ForegroundColor Green }
function Warn  ([string]$m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Fail  ([string]$m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; exit 1 }
function Title ([string]$m) { Write-Host "`n$m" -ForegroundColor Magenta }

Title "================================================="
Title "  LLM Platform Pipeline Runner"
Title "================================================="
Info "Environment : $Env"
Info "Pipeline    : $Pipeline"

# ── Port-forward Kubeflow if no host set ──────────────────────────────────────
if (-not $KubeflowHost) {
    Warn "KUBEFLOW_HOST not set — port-forwarding to localhost:8080..."
    $pf = Start-Process kubectl `
        -ArgumentList "port-forward -n kubeflow svc/ml-pipeline-ui 8080:80" `
        -PassThru -WindowStyle Hidden
    $KubeflowHost = "http://localhost:8080"
    Start-Sleep 4
    Info "Port-forward PID: $($pf.Id)"
}
Info "Kubeflow    : $KubeflowHost"

# ── Helper: watch a pipeline run ─────────────────────────────────────────────
function Watch-Run {
    param([string]$RunId, [int]$TimeoutSec = 120, [string]$Label = "pipeline")
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    Info "Watching $Label run $RunId (timeout: ${TimeoutSec}s)..."
    while ((Get-Date) -lt $deadline) {
        $state = python -c @"
import sys; sys.path.insert(0, '.')
from kfp import client as kfp_sdk
c = kfp_sdk.Client(host='$KubeflowHost')
print(c.get_run(run_id='$RunId').state)
"@ 2>$null
        $elapsed = [int]((Get-Date) - (Get-Date).AddSeconds($TimeoutSec - (($deadline - (Get-Date)).TotalSeconds))).TotalSeconds
        Write-Host "`r  [$Label] $state  " -NoNewline
        if ($state -in @("SUCCEEDED","FAILED","ERROR","SKIPPED")) {
            Write-Host ""
            if ($state -eq "SUCCEEDED") { Ok "$Label SUCCEEDED" }
            else { Fail "$Label ended: $state — check $KubeflowHost" }
            return
        }
        Start-Sleep 15
    }
    Fail "$Label timed out after ${TimeoutSec}s"
}

# ── P05: Data Indexing ────────────────────────────────────────────────────────
function Run-P05 {
    Title "Pipeline 05 — Data Indexing"

    $docText = ""
    if ($DocFile -and (Test-Path $DocFile)) {
        $docText = (Get-Content $DocFile -Raw) -replace "'", "''"
        Info "Loaded: $DocFile"
    } else {
        $docText = "Placeholder document for pipeline testing. Used to validate the P05 data indexing workflow end to end including chunking embedding and upsert to Qdrant."
        if ($DocId) { Warn "No DocFile — using placeholder text for $DocId" }
        else { Fail "Provide -DocId or -DocFile for P05" }
    }
    $eid = if ($DocId) { $DocId } else { "doc-$(Get-Date -Format yyyyMMddHHmmss)" }

    $py = @"
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from kfp import client as kfp_sdk
from pipelines.p05_data_indexing.pipeline import data_indexing_pipeline

c = kfp_sdk.Client(host='$KubeflowHost')
r = c.create_run_from_pipeline_func(
    data_indexing_pipeline,
    arguments={
        'doc_id':     '$eid',
        'source':     '$Source',
        'collection': '$Collection',
        'doc_text':   '$docText',
    },
    run_name='ps-p05-$eid',
    experiment_name='manual-runs',
)
print(r.run_id)
"@
    $run_id = python -c $py
    if ($LASTEXITCODE -ne 0) { Fail "Failed to submit P05" }
    Ok "P05 submitted — run_id: $run_id"
    Info "Monitor: $KubeflowHost/#/runs/details/$run_id"
    Watch-Run -RunId $run_id -TimeoutSec 90 -Label "P05"
}

# ── P09: RAGAS Evaluation ─────────────────────────────────────────────────────
function Run-P09 {
    Title "Pipeline 09 — RAGAS Evaluation"
    Info "qa_limit=$QaLimit, domain='$Domain'"

    $py = @"
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from kfp import client as kfp_sdk
from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline

c = kfp_sdk.Client(host='$KubeflowHost')
r = c.create_run_from_pipeline_func(
    ragas_evaluation_pipeline,
    arguments={
        'qa_limit':      $QaLimit,
        'domain_filter': '$Domain',
        'run_name':      'ps-p09-$(Get-Date -Format yyyyMMddHHmm)',
    },
    run_name='ps-p09-$(Get-Date -Format yyyyMMddHHmmss)',
    experiment_name='manual-runs',
)
print(r.run_id)
"@
    $run_id = python -c $py
    if ($LASTEXITCODE -ne 0) { Fail "Failed to submit P09" }
    Ok "P09 submitted — run_id: $run_id"
    Info "Monitor: $KubeflowHost/#/runs/details/$run_id"
    Info "SLA: up to 30 min for $QaLimit QA pairs"
    Watch-Run -RunId $run_id -TimeoutSec 1800 -Label "P09"

    # Print RAGAS scores from MLflow
    Title "RAGAS Scores (latest P09 run)"
    python -c @"
import sys; sys.path.insert(0, '.')
import mlflow, os
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
runs = mlflow.search_runs(
    experiment_names=['ragas-evaluation'],
    filter_string="tags.pipeline = 'P09-RAGAS'",
    max_results=1, order_by=['start_time DESC'],
)
if runs.empty:
    print('  No P09 runs found in MLflow yet.')
else:
    r = runs.iloc[0]
    metrics = [('faithfulness',0.90,False),('answer_relevancy',0.85,False),
               ('context_precision',0.80,False),('context_recall',0.75,False),
               ('answer_correctness',0.80,False),('hallucination_rate',0.02,True)]
    for m, tgt, lower in metrics:
        val = r.get('metrics.'+m, float('nan'))
        ok  = val <= tgt if lower else val >= tgt
        sym = 'PASS' if ok else 'FAIL'
        dir = '<=' if lower else '>='
        print(f'  [{sym}] {m:<24} {val:.4f}  (target {dir} {tgt})')
    print(f"  Gate: {r.get('tags.gate_passed','unknown')}")
"@
}

# ── Execute ───────────────────────────────────────────────────────────────────
switch ($Pipeline) {
    "p05"  { Run-P05 }
    "p09"  { Run-P09 }
    "both" { Run-P05; Title "P05 done — starting P09 in 5s..."; Start-Sleep 5; Run-P09 }
}

Title "================================================="
Ok "Done"
Title "================================================="
