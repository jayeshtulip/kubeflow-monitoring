# System Design — Enterprise LLM Platform v2.0

*Interview reference document. Covers design decisions, tradeoffs, and key numbers.*

---

## 1. Query Routing — Why 4 Workflows?

The platform routes queries to one of four LangGraph workflows based on a **complexity
score (1–10)** computed from query length, keyword analysis, and entity detection.

| Score | Workflow | Trigger Example | Latency |
|---|---|---|---|
| 1–3 | Simple Research | "What is our Confluence page for EKS alerts?" | ~21s |
| 4–5 | Smart Tools | "Show me recent Jira tickets for payment-svc" | ~33s |
| 6–7 | ReAct | "Why is CPU high on node ip-10-0-1-5?" | ~35s |
| 8–10 | PEC | "Payment service timing out intermittently since 14:20" | ~55s |

**Design decision:** The router saves ~34s on 70% of queries by not invoking the
full PEC loop for simple lookups. Routing overhead is < 50ms.

---

## 2. Planner-Executor-Critic — How the Loop Works

```
User query (complexity ≥ 8)
    │
    ▼
Planner (Mistral-7B)
    Generates 3–7 step investigation plan
    e.g.: ["Check CloudWatch logs 14:00–14:30",
           "Query Jira for recent payment-svc changes",
           "Inspect RDS connection pool metrics"]
    │
    ▼
Executor (tool dispatch loop)
    Calls tools in order: CloudWatch, Jira, Confluence, kubectl, Qdrant
    Builds execution_context with findings
    │
    ▼
Critic (Mistral-7B)
    Verdict: COMPLETE or INCOMPLETE
    If INCOMPLETE: provides specific replan_feedback
    │
    ├── COMPLETE → RAGAS gate → response
    └── INCOMPLETE (max 2 replans) → back to Planner with feedback
```

**Key tradeoffs:**
- Max 2 replans keeps worst-case latency bounded at ~110s
- Critic uses structured output (JSON verdict) to avoid hallucinated "COMPLETE" verdicts
- Tools are read-only whitelisted (`get`, `describe`, `logs`, `top` for kubectl)

---

## 3. RAGAS — Why 6 Metrics, Not Just Accuracy?

BLEU/ROUGE measure n-gram overlap and miss factual grounding entirely.
RAGAS uses NLI-based scoring:

| Metric | What it catches | Example failure |
|---|---|---|
| Faithfulness | Claims not in retrieved context | LLM adds plausible-sounding facts not in docs |
| Answer Relevancy | Off-topic responses | Answering a related but different question |
| Context Precision | Irrelevant chunks retrieved | Top-5 includes unrelated runbooks |
| Context Recall | Missing critical facts | Runbook exists but wasn't retrieved |
| Answer Correctness | Semantic accuracy vs ground truth | Correct format, wrong answer |
| Hallucination Rate | Rate of unsupported claims | Fabricated ticket numbers, port numbers |

**Deployment gate logic:** Faithfulness < 0.85 OR hallucination > 15% → P09 fails →
ArgoCD sync blocked → P11 retraining triggered automatically.

---

## 4. vLLM on H100 — Why Not Just Ollama?

Ollama runs well on CPU/consumer GPU for moderate loads. The H100 adds:

- **PagedAttention** — non-contiguous KV cache eliminates memory waste from fixed-length sequences
- **Continuous batching** — new requests join mid-generation, GPU utilization stays high
- **GPTQ quantization** — Llama-2-7B in 4-bit uses ~4GB VRAM vs 14GB in FP16

At 120 tokens/sec (H100) vs ~15 tokens/sec (CPU Ollama), the H100 serves 8× more
concurrent users at the same latency SLA.

**Traffic split:** Istio routes 70% to Ollama (lower latency for simple queries),
30% to vLLM (higher throughput for batch-heavy PEC workflows). P06 A/B Testing
pipeline adjusts weights weekly based on RAGAS quality + latency T-test.

---

## 5. Kubeflow P11 — Automated Retraining Trigger

Three conditions trigger P11:
1. **RAGAS gate failure** (P09) — faithfulness degraded
2. **Evidently data drift** — JS-divergence > 0.15 on 24h query embedding distribution
3. **Manual trigger** — `run_pipelines.ps1 -Pipeline p11`

P11 steps:
1. `dvc pull` — fetch latest labeled data from S3
2. **Great Expectations gate** — reject if data quality fails (min 20 QA pairs/domain)
3. **Katib HPO** — Bayesian optimization over chunk_size × overlap × top_k
4. **RAGAS evaluation** — new config must beat champion baseline by ≥ 2%
5. **MLflow Registry** → Staging (if pass)
6. **Slack notification** — engineer approves Champion promotion

---

## 6. Data Lineage — Full Trace

Given any live response, you can trace back to the raw source document:

```
Live response
    → pod annotation (model version URI)
    → MLflow Model Registry (version → run_id)
    → MLflow run (run_id → MLMD execution_id)
    → Kubeflow MLMD (execution → DVC SHA)
    → DVC (SHA → S3 path)
    → Raw source document (Confluence page / Jira ticket)
```

Implemented in `mlops/lineage/lineage_tracker.py` as `trace_from_mlflow_run(run_id)`.

---

## 7. Qdrant — Why Not Pinecone/Weaviate?

- **Self-hosted on EKS** — data stays in our VPC, no egress costs, no vendor lock-in
- **StatefulSet + gp3 EBS** — persistent storage survives pod restarts
- **Cosine similarity at 384 dims** — all-MiniLM-L6-v2 embeddings, good balance of quality/speed
- **Multi-collection design** — `tech_docs`, `hr_policies`, `org_info` collections
  allow domain-filtered retrieval, preventing HR policies from contaminating tech incident lookups

---

## 8. Redis — 3 Distinct Uses

1. **Session store** — LangGraph multi-turn conversation state (TTL 1h)
2. **Query dedup cache** — identical queries within 5 min return cached response (TTL 300s)
3. **Rate limiter** — sliding window counter per API key (10 req/min default)

ElastiCache r7g.large (26GB) handles all three with room to spare.

---

## 9. GitHub Actions → ArgoCD Flow

```
PR opened
    └── pr-gate.yml: lint + terraform validate + L1+L2 pytest (< 8 min)

Merge to main
    └── build-push.yml: Docker buildx + Trivy CVE scan + ECR push (commit-SHA tag)
    └── merge-gate.yml: L1+L2+L3 pytest (< 25 min)
    └── ArgoCD Image Updater: detects new ECR tag → updates gitops/apps/dev/

Staging promotion (automatic after dev green)
    └── merge-gate.yml updates gitops/apps/staging/ image tag
    └── ArgoCD syncs staging namespace

Production promotion (manual approval)
    └── GitHub Environments approval gate
    └── release.yml: updates gitops/apps/prod/ image tag
    └── ArgoCD blue/green cutover via Istio VirtualService weight ramp
    └── Automatic rollback if error rate > 1% during ramp
```

---

## 10. Key Interview Numbers to Remember

| Fact | Number |
|---|---|
| EKS workers | 10× c5.4xlarge (16 vCPU, 32GB each) |
| GPU inference | 1× H100 80GB HBM EC2 |
| Qdrant embedding dims | 384 (all-MiniLM-L6-v2) |
| RAGAS golden QA pairs | 60 (20 × tech/hr/org in PostgreSQL) |
| Kubeflow pipelines | 11 (P01–P11) |
| RAGAS faithfulness target | ≥ 0.90 (hard block at 0.85) |
| vLLM throughput | ≥ 80 tokens/sec on H100 |
| Query routing split | 70% simple / 25% ReAct / 5% PEC |
| MTTR improvement | 45 min → 8 min |
| Istio A/B split | Ollama 70% / vLLM 30% |
| Drift threshold | JS-divergence > 0.15 → triggers P11 |
| PEC max replans | 2 (worst-case latency ~110s) |
| Redis TTL (session) | 1 hour |
| Redis TTL (dedup cache) | 5 minutes |
| S3 buckets | 3 (DVC remote, MLflow artifacts, document store) |
| PostgreSQL | RDS Multi-AZ r6g.large, PostgreSQL 15 |
| Redis | ElastiCache r7g.large, 26GB |
