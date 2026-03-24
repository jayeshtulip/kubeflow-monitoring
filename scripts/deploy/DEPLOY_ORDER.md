# Deployment Order — LLM Platform v2.0
**Account:** 659071697671 | **Region:** us-east-1 | **Cluster:** llm-platform-prod

---

## Prerequisites (do once)
```powershell
$env:DB_PASSWORD       = "LLMPlatform2026!"
$env:LANGSMITH_API_KEY = "ls__your_key_from_smith.langchain.com"
$env:HF_TOKEN          = "hf_your_huggingface_token"
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/..."  # optional
$env:GITHUB_ORG        = "your-github-username"
$env:GITHUB_REPO       = "llm-platform"
$env:GITHUB_TOKEN      = "ghp_your_token"
```

---

## Step 1 — Terraform apply (quota approved)
```powershell
cd infra\terraform\environments\prod
terraform apply "tfplan"
# ~15 min — EKS + RDS + Redis + S3
```

---

## Step 2 — Kubeflow + DVC + Grafana
```powershell
cd C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS
.\scripts\deploy\phase2_kubeflow_dvc.ps1
# ~30 min — Kubeflow, Katib, MLflow, Qdrant, Ollama, Prometheus, Grafana
```
**Access after:**
- Kubeflow: `kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow` → http://localhost:8080
- Katib: `kubectl port-forward svc/katib-ui 8081:80 -n kubeflow` → http://localhost:8081
- Grafana: `kubectl port-forward svc/kube-prometheus-grafana 3000:80 -n monitoring` → http://localhost:3000
- MLflow: `kubectl port-forward svc/mlflow 5000:5000 -n mlflow` → http://localhost:5000

---

## Step 3 — vLLM + RAGAS baseline (after GPU quota approved)
```powershell
.\scripts\deploy\phase3_vllm_ragas.ps1
# ~20 min — vLLM, RAGAS baseline evaluation, regression gate test
```

---

## Step 4 — Grafana dashboards + LangSmith + L4 tests
```powershell
.\scripts\deploy\phase4_grafana_langsmith.ps1
# ~20 min — 3 dashboards imported, LangSmith dataset, Locust load test
```

---

## Step 5 — CI/CD + ArgoCD + Istio
```powershell
.\scripts\deploy\phase5_cicd_argocd.ps1
# ~30 min — GitHub OIDC, ArgoCD, Argo Rollouts, Istio, Docker build
```

---

## Final verification (plan: Phase 5 hardening)
```powershell
.\scripts\deploy\post_deploy_verify.ps1
# Runs L1+L2+L3 tests, RAGAS scores, DVC SHA check, Katib status
```

---

## Key pipeline runs to submit (after cluster is up)
```powershell
$env:KUBEFLOW_HOST = "http://localhost:8080"
$env:PYTHONPATH    = (Get-Location).Path

# P05: Index runbooks into Qdrant
.\scripts\mlops\run_pipelines.ps1 -Pipeline p05 -DocFile docs\runbooks\eks_payment_timeout.txt

# P09: RAGAS evaluation (deployment gate)
.\scripts\mlops\run_pipelines.ps1 -Pipeline p09 -QaLimit 60

# P10: DVC reproducibility check
python -c "
from kfp import client as kfp_sdk
from pipelines.p10_dvc_reproducibility.pipeline import dvc_reproducibility_pipeline
c = kfp_sdk.Client(host='http://localhost:8080')
r = c.create_run_from_pipeline_func(dvc_reproducibility_pipeline,
    run_name='dvc-repro-check', experiment_name='manual-runs')
print('P10 submitted:', r.run_id)
"
```

---

## Success criteria (plan: Section 9 + 13)
| Metric | Target | How to verify |
|---|---|---|
| RAGAS faithfulness | >= 0.90 | MLflow → ragas-evaluation experiment |
| Hallucination rate | <= 0.02 | MLflow → ragas-evaluation experiment |
| Simple query P95 | < 30s | Grafana Dashboard 3 |
| PEC query P95 | < 65s | Grafana Dashboard 3 |
| vLLM throughput | >= 80 tok/s | phase3_vllm_ragas.ps1 output |
| Katib HPO | Converged | Katib UI → rag-chunking-experiment |
| DVC SHA | Identical x2 | post_deploy_verify.ps1 |
| L1-L3 tests | All pass | post_deploy_verify.ps1 |
| Grafana MTTD | < 5 min | Alert rules → monitoring namespace |
