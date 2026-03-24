# Enterprise LLM Platform v2.0

**Production-grade multi-agent AI platform for automated incident resolution.**
Built on AWS EKS with LangGraph, Kubeflow, RAGAS, and a full MLOps lifecycle.

---

## Key Metrics

| Metric | Value |
|---|---|
| Query routing latency (simple) | ~21s (Qdrant RAG + Mistral-7B) |
| Query routing latency (complex PEC) | ~55s (Planner-Executor-Critic) |
| RAGAS faithfulness target | ≥ 0.90 |
| Hallucination rate target | ≤ 2% |
| vLLM throughput (H100) | ≥ 80 tokens/second |
| MTTR improvement | 45 min → 8 min (automated resolution) |
| Kubeflow pipelines | 11 (P01–P11) |
| Test coverage | L1×6 + L2×13 + L3×11 + L4×5 = 35 tests |

---

## Architecture

```
User Query
    │
    ▼
FastAPI Gateway  ←── Guardrails (injection/PII/rate-limit)
    │
    ▼
LangGraph Router  ──► complexity score 1-10
    │
    ├── Simple (score ≤ 3) ──────► RAG: Qdrant → Mistral-7B           ~21s
    ├── Smart Tools (4–5) ────────► Keyword tool selection              ~33s
    ├── ReAct (6–7) ─────────────► Think → Act → Observe loop          ~35s
    └── PEC (8–10) ──────────────► Planner → Executor → Critic         ~55s
                                    ├── Tools: CloudWatch, Jira,
                                    │          Confluence, kubectl
                                    └── Max 2 re-plan cycles
    │
    ▼
RAGAS Validation Gate  (faithfulness ≥ 0.85, hallucination ≤ 5%)
    │
    ▼
Response to user  ──► LangSmith trace
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Container orchestration | AWS EKS 1.29 — 10× c5.4xlarge + 1× H100 |
| Multi-agent framework | LangChain 0.2 + LangGraph 0.1 + LangSmith |
| LLM (in-cluster) | Ollama / Mistral-7B |
| LLM (GPU inference) | vLLM 0.4 / Llama-2-7B-GPTQ on H100 (PagedAttention) |
| ML pipelines | Kubeflow Pipelines v2 (KFP SDK 2.4) |
| Hyperparameter optimization | Kubeflow Katib 0.16 |
| RAG evaluation | RAGAS 0.1 (6 metrics) |
| Data versioning | DVC 3.x with S3 remote |
| Vector database | Qdrant 1.8 (384-dim, cosine, EBS-backed) |
| Cache / session | Redis 7 (ElastiCache r7g.large) |
| Relational store | PostgreSQL 15 (RDS Multi-AZ) |
| Experiment tracking | MLflow 2.x (PostgreSQL + S3) |
| Drift detection | Evidently AI 0.4 (JS-divergence) |
| Data quality | Great Expectations 0.18 (3 suites) |
| Observability | LangSmith + Prometheus + Grafana 10 |
| CI/CD | GitHub Actions + ArgoCD 2.x (app-of-apps) |
| Service mesh | Istio 1.20 (blue/green + A/B traffic) |
| IaC | Terraform 1.7 |
| Secrets | AWS Secrets Manager + External Secrets Operator |
| Testing | pytest 8 + Locust 2.28 |

---

## Project Structure

```
RAGAS-KUBEFLOW-MLOPS/
├── src/
│   ├── api/                   FastAPI gateway + guardrails
│   ├── agents/                Planner, Executor, Critic, Router + 5 tools
│   ├── workflows/             4 LangGraph workflows (simple/react/smart/PEC)
│   ├── storage/               Qdrant indexer/retriever, Redis sessions, PostgreSQL
│   ├── serving/               Ollama client, vLLM client
│   ├── ragas_eval/            RAGAS evaluator + golden QA dataset builder
│   ├── drift/                 Evidently drift monitor
│   └── observability/         LangSmith tracer, Prometheus metrics, Grafana exporter
├── pipelines/
│   ├── p01_model_evaluation/  Daily: 4 workflows × 5 queries
│   ├── p02_rag_optimization/  Weekly: 9 chunking strategies
│   ├── p03_prompt_engineering/On model update: 18 templates
│   ├── p04_quality_monitoring/Every 15 min: HEALTHY/DEGRADED/CRITICAL
│   ├── p05_data_indexing/     On upload: chunk → embed → Qdrant
│   ├── p06_ab_testing/        Weekly: ReAct vs PEC T-test
│   ├── p07_hallucination/     Post-deploy: factual grounding 100 queries
│   ├── p08_guardrail/         Weekly: injection + PII detection
│   ├── p09_ragas_evaluation/  Post-staging: 6 metrics → deployment gate
│   ├── p10_dvc_reproducibility/Weekly: dvc repro × 2, compare SHAs
│   ├── p11_automated_retraining/Drift alert: GX → Katib HPO → RAGAS → Registry
│   └── components/            5 reusable KFP components
├── tests/
│   ├── l1_infrastructure/     6 tests: EKS nodes, H100, RDS, Redis, S3
│   ├── l2_component/          13 tests: Qdrant, vLLM, Redis, PostgreSQL, RAGAS
│   ├── l3_integration/        11 tests: E2E workflows, P05/P09 pipelines, LangSmith
│   └── l4_performance/        5 tests: Locust load, vLLM throughput
├── infra/
│   ├── terraform/             EKS, RDS, ElastiCache, ECR, IAM modules
│   ├── helm/llm-platform/     Helm chart with 6 templates + HPA
│   └── k8s/                   Qdrant, Ollama, MLflow manifests + Istio routing
├── monitoring/
│   ├── grafana/dashboards/    3 JSON dashboards (Data Health, MLflow, Performance)
│   └── prometheus/rules/      Platform alerts + SLA burn rate
├── data/
│   ├── golden_qa/             60 QA pairs (tech/hr/org — seeded in PostgreSQL)
│   └── great_expectations/    3 validation suites
├── mlops/
│   ├── dvc/params/            rag_params.yaml (chunk_size, overlap, top_k, XGBoost)
│   ├── model_registry/        Staging → Champion promotion scripts
│   └── lineage/               Full lineage tracker (response → raw data)
├── .github/workflows/         5 CI/CD workflows (PR gate, build-push, merge, nightly, TF plan)
├── gitops/                    ArgoCD app-of-apps for dev/staging/prod
├── docker/                    Dockerfiles for api, agents, vLLM (H100)
├── dvc.yaml                   5 reproducible DVC stages
└── conftest.py                pytest root conftest (6 shared fixtures)
```

---

## Quick Start (Local Dev)

```powershell
# 1. Bootstrap tools and install Python packages
.\scripts\setup\bootstrap.ps1

# 2. Start local services (PostgreSQL, Redis, Qdrant, MLflow)
docker-compose -f docker-compose.dev.yml up -d

# 3. Seed golden QA data
python scripts/data/seed_golden_qa.py

# 4. Verify imports
python -c "from src.api.main import app; print('API OK')"
python -c "from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline; print('P09 OK')"

# 5. Run L1 + L2 tests (local mode, no AWS needed)
pytest tests/ -m "not aws and not live" -v
```

---

## Deploying to AWS

```powershell
# Provision infrastructure
cd infra/terraform/environments/prod
terraform init
terraform plan
terraform apply

# Deploy platform via ArgoCD
kubectl apply -f infra/k8s/namespaces.yaml
helm upgrade --install llm-platform infra/helm/llm-platform/ \
    -f infra/helm/llm-platform/values-prod.yaml \
    --namespace llm-platform-prod

# Seed production PostgreSQL
$env:POSTGRES_HOST = "<rds-endpoint>"
python scripts/data/seed_golden_qa.py

# Run P05 to index documents
.\scripts\mlops\run_pipelines.ps1 -Pipeline p05 -DocFile docs\runbooks\eks_incidents.txt

# Run P09 RAGAS evaluation (deployment gate)
.\scripts\mlops\run_pipelines.ps1 -Pipeline p09 -QaLimit 200

# Check platform health
.\scripts\debug\check_platform_health.ps1
```

---

## RAGAS Evaluation Thresholds

| Metric | Target | Hard Block |
|---|---|---|
| Faithfulness | ≥ 0.90 | < 0.85 → blocks deploy, triggers P11 |
| Answer Relevancy | ≥ 0.85 | — |
| Context Precision | ≥ 0.80 | — |
| Context Recall | ≥ 0.75 | — |
| Answer Correctness | ≥ 0.80 | — |
| Hallucination Rate | ≤ 0.02 | > 0.15 → blocks deploy, triggers P11 |

---

## MLOps Lifecycle

```
Raw Data (Confluence/Jira/upload)
    │ Great Expectations gate
    ▼
DVC commits data SHA to S3 remote
    │ dvc repro (P10 weekly)
    ▼
Kubeflow P05: chunk → embed → Qdrant
    │ P02 RAG optimization + Katib HPO
    ▼
Kubeflow P09: RAGAS evaluation
    │ faithfulness ≥ 0.85?
    ├── YES → MLflow Registry: Staging stage
    │          ArgoCD deploys to staging
    │          L3 tests + manual approval
    │          Production blue/green cutover (Istio)
    └── NO  → Kubeflow P11: Automated retraining
                Evidently drift → Katib HPO → P09 re-runs
```

---

## Grafana Dashboards

1. **Data Health** — DVC commit age, JS-divergence drift score, QA coverage heatmap, Qdrant index freshness
2. **MLflow Metrics** — RAGAS scores over time, workflow latency P50/P95/P99, Katib HPO convergence, A/B test results
3. **Platform Performance** — Request throughput, workflow routing split, vLLM queue depth, Redis hit rate, PostgreSQL connections

---

## Prometheus Alerts

- `CriticalP95Latency` — P95 > 65s for 2 min → PagerDuty page
- `CriticalSuccessRate` — Success rate < 85% for 2 min → PagerDuty page
- `HallucinationRateCritical` — Rate > 5% for 5 min → ML team alert
- `DataDriftDetected` — JS-divergence > 0.15 → triggers P11 retraining
- `SLABurnRateFast` — Error budget burning at 14.4× → PagerDuty page

---

## Author

Jayesh | Senior AI/ML Validation & MLOps Engineer  
15+ years: AI/ML validation, MLOps, Cloud Infrastructure, Networking  
Zadara Cloud Technologies → Cisco → Wipro
