# Interview Preparation — Enterprise LLM Platform v2.0

*Comprehensive Q&A for Senior AI/ML Platform Engineer interviews.*
*Cover every section before technical interviews at AI Platform / AIOps companies.*

---

## SECTION 1: Project Overview (Opening Questions)

**Q: Walk me through this project.**

This is a production MLOps platform on AWS EKS that automatically resolves infrastructure
incidents using a multi-agent AI system. When an engineer raises a query — like "why is
the payment service timing out?" — the platform routes it through LangGraph workflows,
uses LangChain agents to query CloudWatch, Jira, Confluence, and kubectl, then generates
a root-cause analysis grounded in our internal runbooks.

Key numbers: 21–55s response latency depending on complexity, MTTR improved from 45 minutes
to 8 minutes, faithfulness ≥ 0.90 via RAGAS evaluation, 11 Kubeflow pipelines for
continuous evaluation and retraining.

---

**Q: Why build this vs using an off-the-shelf solution like ServiceNow AI or PagerDuty AIOps?**

Our incident data is proprietary — CloudWatch logs, internal runbooks, architecture docs.
Off-the-shelf solutions can't be trained on that. By building it ourselves we get full
data lineage (every response traceable to source documents), domain-specific RAG tuning
via Katib HPO, and the ability to gate deployments on RAGAS quality scores — none of
which commercial tools offer at this level of control.

---

**Q: What was the hardest engineering problem?**

The RAGAS evaluation pipeline gating CI/CD deployments. The challenge was making P09
a true deployment gate — not just a reporting tool. We had to wire ArgoCD sync hooks
to the Kubeflow pipeline outcome, so a faithfulness drop below 0.85 actually blocks the
deployment and triggers automated retraining (P11), all without human intervention.
Getting the KFP `dsl.If` branching correct with `.outputs['Output']` syntax took
significant debugging.

---

## SECTION 2: LangGraph & Multi-Agent Architecture

**Q: Why LangGraph over vanilla LangChain AgentExecutor?**

LangGraph gives us a directed graph with explicit state management. With vanilla
AgentExecutor you get an opaque loop — hard to add conditional branching, retry logic,
or max-iteration limits. With LangGraph we explicitly define the Planner → Executor →
Critic edges, can branch on Critic verdict (COMPLETE vs INCOMPLETE), and cap replans
at 2 to bound worst-case latency. State is serializable to Redis for session continuity
across multi-turn conversations.

---

**Q: Explain the Planner-Executor-Critic pattern.**

Planner (Mistral-7B via Ollama) receives the user query and generates a structured
investigation plan — 3 to 7 steps like "Check CloudWatch logs 14:00–14:30 for payment-svc"
and "Query Jira for recent deployments in the payment namespace."

Executor dispatches those steps as tool calls — CloudWatch Logs Insights queries,
Jira JQL searches, Confluence CQL searches, read-only kubectl commands — and accumulates
findings in an execution context.

Critic receives the original query, the plan, and all findings, then returns a structured
JSON verdict: COMPLETE with a final answer, or INCOMPLETE with specific feedback on what's
missing. If INCOMPLETE, the feedback goes back to the Planner for a revised plan. Max 2
replans to prevent infinite loops.

---

**Q: How do you prevent the agents from taking destructive actions?**

kubectl tool has a strict whitelist: only `get`, `describe`, `logs`, and `top` commands.
Any other kubectl verb raises a SecurityError before execution. All external API calls
are read-only (Jira read-only token, Confluence read-only token). There's no
write path in any tool. This is enforced in `src/agents/tools/kubectl.py` at the
function level, not just by prompt engineering.

---

**Q: What is complexity scoring and how does it work?**

The WorkflowRouter computes a score 1–10 for each incoming query using:
- Query length (longer = more complex)
- Presence of temporal markers ("since", "intermittently", "after deploy")
- Multi-entity detection (multiple service names, multiple symptoms)
- Negation and comparison patterns ("not working", "slower than")

Score ≤ 3 → Simple Research (Qdrant RAG only)
Score 4–5 → Smart Tools (keyword-based tool selection)
Score 6–7 → ReAct (Think-Act-Observe, max 5 steps)
Score 8–10 → Planner-Executor-Critic (full investigation)

The router adds < 50ms overhead and saves 34 seconds on 70% of queries.

---

## SECTION 3: RAG & Vector Database

**Q: Why sentence-transformers all-MiniLM-L6-v2 specifically?**

It's the sweet spot for our use case: 384 dimensions (small enough for fast cosine
search in Qdrant), 80ms inference per query on CPU, and strong performance on
technical text (Stack Overflow, GitHub Issues are in its training data). We evaluated
it against `bge-small-en` and `e5-small-v2` — MiniLM-L6 had 8% better context recall
on our golden QA dataset at similar speed.

---

**Q: Explain your chunking strategy.**

Fixed-size overlapping windows: 150 tokens per chunk, 30-token overlap. The overlap
ensures sentences at chunk boundaries aren't split mid-context. Chunk size and overlap
are DVC-tracked parameters in `mlops/dvc/params/rag_params.yaml` and tuned weekly by
Kubeflow P02 RAG Optimization pipeline using Katib Bayesian optimization.
The objective metric is RAGAS Context Recall — we want retrieved chunks to cover
ground-truth facts, not just be semantically similar to the question.

---

**Q: How do you handle multi-collection search in Qdrant?**

We have three collections: `tech_docs` (runbooks, architecture docs, incident reports),
`hr_policies`, and `org_info`. The `multi_collection_search` function in
`src/storage/qdrant/retriever.py` queries all relevant collections in parallel and
merges results by cosine score. Domain filtering prevents HR policy chunks from
appearing in tech incident responses — the collections are queried selectively based
on query intent detected by the router.

---

**Q: What are the RAGAS metrics and why use them over BLEU/ROUGE?**

BLEU and ROUGE measure n-gram overlap against a reference — they'll score "the server
is down" and "the database is down" the same if the words overlap. RAGAS uses NLI-based
scoring:

- **Faithfulness**: Are the claims in the response supported by the retrieved context?
  Catches hallucinations — the LLM adding plausible-sounding facts not in our runbooks.
- **Answer Relevancy**: Is the response actually answering the question asked?
- **Context Precision**: Of the top-5 chunks retrieved, how many were actually relevant?
  Low precision = the retriever is pulling noise.
- **Context Recall**: Of the ground-truth facts, how many appear in the retrieved chunks?
  Low recall = important runbook sections are being missed.
- **Answer Correctness**: Combined factual + semantic accuracy vs ground truth.
- **Hallucination Rate**: Fraction of response claims absent from source documents.

We gate deployment on Faithfulness ≥ 0.85 and Hallucination Rate ≤ 15%.

---

## SECTION 4: MLOps & Kubeflow

**Q: Walk me through the 11 Kubeflow pipelines.**

- P01 Model Evaluation — daily, tests all 4 workflows on 5 queries each, logs latency and quality to MLflow
- P02 RAG Optimization — weekly, tests 9 chunking strategies, picks best by Context Recall
- P03 Prompt Engineering — on model update, tests 18 prompt templates × 3 agents
- P04 Quality Monitoring — every 15 min, determines HEALTHY/DEGRADED/CRITICAL platform state
- P05 Data Indexing — on document upload, runs GX validation → DVC → chunk → embed → Qdrant
- P06 A/B Testing — weekly, T-test on ReAct vs PEC quality scores and latency
- P07 Hallucination Testing — post-deploy, runs 100 queries against known facts
- P08 Guardrail Effectiveness — weekly, tests injection patterns and PII detection
- P09 RAGAS Evaluation — post-staging deploy, 6-metric gate, blocks or promotes to MLflow Registry
- P10 DVC Reproducibility — weekly, runs `dvc repro` twice, compares output SHAs
- P11 Automated Retraining — triggered by drift alert or P09 failure, runs GX → Katib → RAGAS → Registry

---

**Q: How does Katib hyperparameter optimization work in your system?**

Katib is Kubeflow's HPO component. We define a `katib/experiments/rag_chunking_experiment.yaml`
with the search space: `chunk_size` (50–200), `overlap` (10–50), `top_k` (3–10).
The objective metric is `ragas_faithfulness` from the RAGAS evaluation run.

Katib uses Bayesian optimization — it models the objective function using a Gaussian
process and picks the next trial to maximize expected improvement over the current best.
Each trial runs the full preprocess → embed → index → RAGAS pipeline. After convergence
(typically 20–30 trials), Katib logs the best hyperparameters to MLflow and those values
update `rag_params.yaml` via a DVC commit.

---

**Q: Explain DVC and why you're using it.**

DVC (Data Version Control) treats data the same way Git treats code. Every dataset,
embedding file, and model checkpoint gets a `.dvc` pointer file committed to Git,
while the actual binary data lives in S3. The SHA in the pointer file uniquely identifies
the exact data version.

Key benefits in our system:
1. **Reproducibility** — P10 pipeline runs `dvc repro` twice and compares output SHAs.
   If they differ, the pipeline raises an alert (non-determinism in data processing).
2. **Lineage** — you can trace any production response back to the exact DVC SHA that
   produced the Qdrant collection serving it.
3. **Rollback** — if retraining degrades quality, `dvc checkout <sha>` restores the
   previous data version and re-indexes Qdrant.

---

**Q: How does the MLflow Model Registry promotion work?**

Four stages: None → Staging → Champion (Production) → Archived.

None: any MLflow run. An engineer nominates it to Staging by tagging it.
Staging: P09 RAGAS gate must pass (faithfulness ≥ 0.85, hallucination ≤ 15%).
  If pass → `client.transition_model_version_stage(name, version, 'Staging')`.
Champion: Senior engineer manually approves in GitHub Environments UI.
  ArgoCD Image Updater picks up the new model reference and updates the GitOps manifest.
  Istio ramps traffic 10% → 50% → 100% to the new version over 10 minutes.
  Auto-rollback if error rate > 1% during ramp.
Archived: when a new Champion is promoted, the previous one is archived but kept in
  S3 for 90 days.

---

## SECTION 5: Infrastructure & CI/CD

**Q: Walk me through the CI/CD pipeline from PR to production.**

PR opened → `pr-gate.yml`: ruff lint, terraform validate, L1+L2 pytest (< 8 min).
Merge to main → `build-push.yml`: Docker buildx multi-arch, Trivy CVE scan (exits on
HIGH/CRITICAL), ECR push with commit-SHA tag. Then `merge-gate.yml`: L1+L2+L3 pytest
(< 25 min). ArgoCD Image Updater detects the new ECR tag and auto-syncs the dev namespace.

Staging: after dev is green and L3 passes, merge-gate.yml updates the staging GitOps
manifest. ArgoCD syncs staging. P09 RAGAS runs against staging.

Production: manual approval in GitHub Environments. `release.yml` updates prod GitOps
manifest. ArgoCD blue/green cutover — Istio VirtualService shifts traffic from blue
(old) to green (new) in a canary ramp. Automated rollback via Argo Rollouts if Prometheus
alerts fire during the ramp.

---

**Q: Why ArgoCD over Flux or plain Helm?**

ArgoCD's app-of-apps pattern lets us manage multiple environments (dev/staging/prod) as
separate ArgoCD Application resources from one root Application. The GitOps model means
every state change is a Git commit — full audit trail, instant rollback via `git revert`.
ArgoCD Image Updater handles the ECR tag update automatically, so the CI/CD loop is:
code push → ECR push → Image Updater commits tag update → ArgoCD syncs. No manual
kubectl apply anywhere in the production path.

---

**Q: How does the blue/green deployment work with Istio?**

Two Kubernetes Deployments co-exist: `blue` (current production) and `green` (new version).
An Istio VirtualService weight controls traffic split. Initially: blue=100%, green=0%.

When a new version is deployed:
1. `green` Deployment is updated with the new image
2. Argo Rollouts gradually shifts weights: 10% → 25% → 50% → 100% over 10 minutes
3. Prometheus monitors error rate and P95 latency during ramp
4. If error rate > 1% → Argo Rollouts immediately reverts to blue=100% (no pod restart needed)
5. If ramp completes successfully → blue Deployment is updated to match green for next cycle

Zero downtime because traffic is shifted at the Istio layer, not by pod restarts.

---

**Q: How is secrets management handled?**

No secrets in Git, ever. The flow:
1. Secrets live in AWS Secrets Manager (RDS password, LangSmith API key, etc.)
2. External Secrets Operator runs in the cluster, watches SecretStore CRDs
3. It creates Kubernetes Secret objects synced from Secrets Manager every hour
4. Pods reference those Kubernetes Secrets via `secretKeyRef`
5. IRSA (IAM Roles for Service Accounts) gives each service a dedicated IAM role
   with least-privilege read access to only its own secrets
6. Rotation: RDS credentials rotate automatically every 30 days via Secrets Manager
   rotation Lambda; ESO picks up new values within one hour

---

## SECTION 6: Observability

**Q: What's in the three Grafana dashboards?**

Dashboard 1 — **Data Health**: DVC last commit timestamp, JS-divergence drift score
(rolling 7-day), golden QA coverage heatmap by domain, Qdrant collection vector count
and index freshness, DVC pipeline stage durations.

Dashboard 2 — **MLflow Metrics**: RAGAS scores over time (faithfulness, relevancy),
hallucination rate trend, workflow latency P50/P95/P99 by workflow type, Katib HPO
convergence scatter plot, A/B test results from P06.

Dashboard 3 — **Platform Performance**: request throughput (req/min), workflow routing
split (simple/ReAct/PEC percentages), vLLM generation queue depth and tokens/sec,
EKS node CPU/memory, Redis hit rate and memory, Qdrant query latency P95,
guardrail trigger frequency by type, PostgreSQL connection pool utilization.

---

**Q: How does Evidently drift detection work?**

Evidently runs every 6 hours as a Kubeflow pipeline component. It computes the
Jensen-Shannon divergence between:
- Baseline: embedding distribution of the original training/indexing corpus
- Current: embedding distribution of the last 24 hours of production queries

JS-divergence > 0.15 means the incoming query distribution has shifted significantly
from what the RAG system was optimized for — new terminology, new incident types,
new service names. This triggers a Prometheus alert which fires a webhook to start P11
automated retraining.

We also monitor concept drift: if the rolling 7-day RAGAS faithfulness drops more than
10% relative to the champion baseline, that's concept drift (the LLM's responses are
degrading even if the queries look the same).

---

**Q: What does LangSmith give you that Prometheus doesn't?**

Prometheus gives you aggregated metrics — latency histograms, error rates, throughput.
LangSmith gives you **trace-level observability** into every agent decision:

- Which step of the Planner's plan was executed
- What each tool call returned
- Whether the Critic's verdict was COMPLETE or INCOMPLETE and why
- Token usage per agent step
- Human feedback (thumbs up/down) correlated with trace IDs

When a user reports a bad response, we can pull the LangSmith trace and replay every
decision that led to it. You can't do that with Prometheus.

---

## SECTION 7: vLLM & GPU Inference

**Q: What is PagedAttention and why does it matter?**

Standard transformer inference pre-allocates a contiguous KV cache block for each
request based on the maximum sequence length. If a request only uses 200 of 4096 tokens,
the remaining 3896 tokens of KV cache are wasted — internal fragmentation.

PagedAttention (from the vLLM paper) uses non-contiguous virtual memory for KV cache —
like OS virtual memory management. KV cache blocks are allocated on demand in pages,
and a block table maps logical KV positions to physical memory. This eliminates internal
fragmentation, allowing vLLM to serve 3–4× more concurrent requests at the same latency
on H100 compared to Hugging Face standard inference.

---

**Q: Why GPTQ quantization for Llama-2-7B?**

GPTQ (Generative Pre-trained Transformer Quantization) post-trains a 4-bit integer
quantization of the model weights. Llama-2-7B in FP16 needs ~14GB VRAM; in GPTQ 4-bit
it needs ~4GB. On the H100 80GB this lets us run the model with much larger batch sizes —
more concurrent requests per GPU, higher throughput. GPTQ has minimal quality degradation
compared to FP16 on instruction-following tasks (< 2% perplexity increase on standard benchmarks).

---

## SECTION 8: Testing Strategy

**Q: How do you test a system that depends on LLM outputs?**

Four-layer approach:

L1 (Infrastructure): deterministic — assert EKS nodes are Ready, H100 GPU is
allocatable, RDS latency < 50ms, S3 buckets accessible. No LLM involvement.

L2 (Component): semi-deterministic — Qdrant upsert and query (cosine ≥ 0.70 threshold),
Redis set/get/delete, vLLM health endpoint, RAGAS on 10 fixed samples with known answers.
We set RAGAS faithfulness threshold at 0.70 for L2 (lower than production 0.85) because
we're using a small sample.

L3 (Integration): statistical — submit real queries through the full stack and assert
routing, latency SLAs, and that LangSmith traces are recorded with the right structure.
We use fixed "golden queries" with known expected routing (e.g. "What is our EKS cluster
name?" must route to Simple, not PEC).

L4 (Performance): Locust load test at 50 req/min sustained for 10 minutes, assert
P95 latency < 60s. vLLM throughput test asserts ≥ 80 tokens/sec on H100.

---

**Q: How do you handle flaky tests from non-deterministic LLM outputs?**

Three strategies:
1. **Routing tests use routing assertions, not content assertions.** We check that
   "payment service timeout" routes to PEC (score ≥ 8) — not that the answer contains
   specific words.
2. **RAGAS thresholds have production vs test slack.** Production gate is faithfulness
   ≥ 0.85; L2 component tests use 0.70 to allow for small sample variance.
3. **LangSmith trace assertions are structural, not content-based.** We assert that
   the PEC workflow produced a Planner trace, an Executor trace, and a Critic trace with
   a `verdict` field — not what those traces contain.

---

## SECTION 9: Behavioural / Design Questions

**Q: A RAGAS faithfulness score drops from 0.92 to 0.81 in production. What do you do?**

1. Check Grafana Data Health dashboard — has the Qdrant collection size changed? Did a
   recent P05 indexing run add bad documents?
2. Check Evidently drift score — is the query distribution shifting? New incident types
   the system wasn't trained on?
3. Check P07 Hallucination Testing pipeline results — is this isolated to specific
   domains or widespread?
4. Check LangSmith traces from the time of the drop — are agents making poor tool calls
   or the retriever returning irrelevant chunks?
5. If drift is confirmed → manually trigger P11 retraining.
6. If data quality is suspected → run Great Expectations on recent indexed documents,
   check the P05 rejection rate.
7. If neither → rollback to previous MLflow Champion version using ArgoCD + Istio
   (shift traffic back to blue in < 60 seconds).

---

**Q: How would you scale this to 10× the query volume?**

Current bottleneck at 10× load is likely the Ollama inference layer (CPU-bound).
Scale path:
1. **Short term**: Increase Ollama deployment replicas (HPA already configured, targets
   65% CPU). Add more c5.4xlarge EKS nodes via ASG scaling.
2. **Medium term**: Shift more traffic to vLLM on H100 (change Istio weights to 30/70
   Ollama/vLLM). Add a second H100 instance.
3. **Long term**: Move Ollama to GPU inference (A10G instances). Implement request
   deduplication cache in Redis to avoid re-running identical queries (already have the
   Redis dedup cache, just need to tune TTL).
4. **Redis session store**: already using ElastiCache r7g.large — add read replicas if
   session reads become the bottleneck.
5. **Qdrant**: already on StatefulSet with gp3 EBS — migrate to Qdrant Cloud or add
   horizontal sharding across collections.

---

**Q: What would you do differently if building this from scratch today?**

1. **Use KFP SDK v2.7+ from the start** — we had to pin to 2.4 due to strict artifact
   validation changes. The newer SDK has better local debugging tools.
2. **Start with LangSmith dataset-driven evaluation earlier** — we built custom RAGAS
   pipelines before realizing LangSmith Evaluators can do much of the same work with
   less infrastructure overhead.
3. **Istio from day one** — retrofitting the A/B traffic split after the initial deploy
   required rewriting Kubernetes Services and adding DestinationRules. Much easier to
   design it in upfront.
4. **External Secrets Operator from day one** — we had environment variables in
   ConfigMaps initially, which is a security risk. ESO should be a prerequisite, not
   an afterthought.

---

## KEY NUMBERS QUICK REFERENCE

| Metric | Value |
|---|---|
| Simple query latency | ~21s |
| ReAct latency | ~35s |
| PEC latency | ~55s |
| PEC max replans | 2 |
| RAGAS faithfulness target | ≥ 0.90 |
| RAGAS hard block | < 0.85 |
| Hallucination rate target | ≤ 2% |
| vLLM throughput | ≥ 80 tok/s |
| MTTR improvement | 45 min → 8 min |
| EKS workers | 10× c5.4xlarge |
| H100 VRAM | 80GB |
| Qdrant dims | 384 |
| Istio A/B split | Ollama 70% / vLLM 30% |
| Drift threshold | JS-div > 0.15 |
| Golden QA pairs | 60 (20 per domain) |
| Kubeflow pipelines | 11 |
| Test functions | 35 (L1×6 + L2×13 + L3×11 + L4×5) |
| Redis session TTL | 1 hour |
| Redis dedup TTL | 5 minutes |
| Secrets rotation | 30 days (auto) |
| GX min pairs/domain | 20 |
| Katib trials (typical) | 20–30 |

---

## COMMON FOLLOW-UP DEPTH TESTS

- "Show me the code for the Critic's verdict parsing" → `src/agents/critic/critic_agent.py`
- "How does the guardrail detect prompt injection?" → `src/guardrails/input_validator.py` — 10 regex patterns
- "What happens if Qdrant is down?" → `/ready` endpoint returns 503, router falls back to LLM-only mode
- "How do you prevent the Executor from running the same tool twice?" → execution_context deduplication in `executor_agent.py`
- "What's the KFP component base image?" → `python:3.11-slim` with per-component `packages_to_install`
- "Why not use Triton for Mistral-7B?" → Triton requires model files in repository format; Ollama provides a simpler operational model for dynamic model loading
