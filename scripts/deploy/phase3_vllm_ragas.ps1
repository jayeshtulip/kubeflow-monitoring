<#
.SYNOPSIS
    Phase 3: vLLM on g4dn.2xlarge + RAGAS Evaluation
    Run AFTER GPU quota is approved.

    Plan: Section 3 + Phase 3 (Weeks 5-6)
    - Deploys vLLM with Mistral-7B-Instruct-GPTQ-4bit on T4 GPU
    - Runs first RAGAS baseline evaluation via Kubeflow P09
    - Validates regression gate (intentional prompt degradation test)
    - Logs baseline scores to MLflow experiment 'ragas-evaluation'
#>

$ErrorActionPreference = "Stop"
$CLUSTER   = "llm-platform-prod"
$REGION    = "us-east-1"
$NAMESPACE = "llm-platform-prod"
$DEST      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Log  ([string]$m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok   ([string]$m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Title([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Magenta }

# ── Step 1: Scale up GPU node ─────────────────────────────────────────────────
Title "Step 1: Scale up g4dn.2xlarge GPU node"
aws eks update-nodegroup-config `
  --cluster-name $CLUSTER `
  --nodegroup-name gpu-worker `
  --scaling-config minSize=0,maxSize=1,desiredSize=1 `
  --region $REGION

Log "Waiting for GPU node to join cluster (~3 min)..."
Start-Sleep 30
kubectl wait --for=condition=Ready node -l role=gpu --timeout=300s
kubectl get nodes -l role=gpu
Ok "GPU node ready"

# ── Step 2: Deploy vLLM (plan: Section 2.2 — Mistral-7B-GPTQ on T4) ─────────
Title "Step 2: Deploy vLLM (Mistral-7B-Instruct-GPTQ-4bit)"
Log "Model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ (~4GB VRAM on T4 16GB)"

# Create HuggingFace token secret
kubectl create secret generic llm-platform-secrets `
  --namespace $NAMESPACE `
  --from-literal=hf-token=$env:HF_TOKEN `
  --from-literal=postgres-password=$env:DB_PASSWORD `
  --from-literal=langsmith-api-key=$env:LANGSMITH_API_KEY `
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$DEST\infra\k8s\vllm\vllm-deployment.yaml"
Log "Waiting for vLLM to load model (~3 min)..."
kubectl rollout status deployment/vllm-server -n $NAMESPACE --timeout=300s
Ok "vLLM deployed — Mistral-7B-Instruct-GPTQ-4bit on T4"

# ── Step 3: Test vLLM throughput (plan: Section 9 — >= 80 tok/s) ─────────────
Title "Step 3: L2 vLLM throughput test (plan: >= 80 tok/s)"
kubectl port-forward svc/vllm-service 8000:8000 -n $NAMESPACE &
Start-Sleep 5

python "$DEST\scripts\validate\test_vllm_throughput.py"
Ok "vLLM throughput test complete"

# ── Step 4: Run RAGAS baseline evaluation (plan: Section 3.2, Phase 3) ────────
Title "Step 4: RAGAS baseline evaluation (Kubeflow P09)"
Log "Submitting P09 RAGAS Evaluation pipeline..."

kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow &
Start-Sleep 5

$env:PYTHONPATH = $DEST
$env:KUBEFLOW_HOST = "http://localhost:8080"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"  # port-forwarded

# Port-forward MLflow too
kubectl port-forward svc/mlflow 5000:5000 -n mlflow &
Start-Sleep 3

python -c "
import sys, os
sys.path.insert(0, os.environ['PYTHONPATH'])
from kfp import client as kfp_sdk
from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline

c = kfp_sdk.Client(host='http://localhost:8080')
run = c.create_run_from_pipeline_func(
    ragas_evaluation_pipeline,
    arguments={
        'qa_limit':             60,
        'domain_filter':        '',
        'run_name':             'baseline-ragas-evaluation',
        'mlflow_tracking_uri':  'http://mlflow-service.mlflow.svc.cluster.local:5000',
    },
    run_name='phase3-baseline-ragas',
    experiment_name='ragas-evaluation',
)
print(f'P09 submitted: run_id={run.run_id}')
print(f'Monitor: http://localhost:8080/#/runs/details/{run.run_id}')
"
Ok "P09 RAGAS baseline submitted — check Kubeflow UI"

# ── Step 5: Regression gate validation (plan: Phase 3 — intentional degradation)
Title "Step 5: Regression gate test (plan: Phase 3)"
Log "Testing that P09 blocks deployment on degraded prompts..."
python -c "
from src.ragas_eval.evaluator import _check_gate

# Simulate a degraded model (faithfulness 0.60 — below 0.85 hard block)
scores = {'faithfulness': 0.60, 'hallucination_rate': 0.03}
passed, failures = _check_gate(scores)
print(f'  Degraded model gate result: passed={passed}')
print(f'  Failures: {failures}')
assert not passed, 'Gate should have failed for faithfulness=0.60'

# Good model
good_scores = {'faithfulness': 0.92, 'hallucination_rate': 0.01}
passed2, _ = _check_gate(good_scores)
assert passed2, 'Gate should pass for faithfulness=0.92'
print(f'  Good model gate result: passed={passed2}')
print('  Regression gate works correctly')
"
Ok "Regression gate validated — blocks deploy on faithfulness < 0.85"

Title "Phase 3 Complete"
Write-Host "`n  vLLM serving Mistral-7B-GPTQ on T4 GPU" -ForegroundColor Green
Write-Host "  RAGAS baseline running in Kubeflow — check MLflow for scores" -ForegroundColor Green
Write-Host "  Next: Run phase4_grafana_langsmith.ps1" -ForegroundColor Cyan
