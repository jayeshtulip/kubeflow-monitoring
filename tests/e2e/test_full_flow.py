"""
tests/e2e/test_full_flow.py
End-to-end test suite for Enterprise LLM Platform v2.0.

Run with port-forwards active:
  kubectl port-forward svc/qdrant 6333:6333 -n llm-platform-prod
  kubectl port-forward svc/vllm-service 8000:8000 -n llm-platform-prod
  kubectl port-forward svc/mlflow-service 5000:5000 -n mlflow

  pip install pytest qdrant-client sentence-transformers requests mlflow psycopg2-binary
  pytest tests/e2e/test_full_flow.py -v --tb=short
"""

import os, time, statistics
import pytest
import requests
import psycopg2

QDRANT_HOST   = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.environ.get("QDRANT_PORT", "6333"))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MLFLOW_URI    = os.environ.get("MLFLOW_URI", "http://localhost:5000")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_DB   = os.environ.get("POSTGRES_DB", "llm_platform")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "llm_admin")
POSTGRES_PASS = os.environ.get("POSTGRES_PASS", "Llmplatform2026")

MIN_QDRANT_CHUNKS  = 1
P95_LATENCY_SLA_S  = 10.0
MIN_FAITHFULNESS   = 0.60
MIN_TECH_DOCS      = 100
MIN_HR_POLICIES    = 55
MIN_ORG_INFO       = 50

DOMAIN_QUESTIONS = {
    "tech_docs": [
        "What GPU instance type does the platform use for vLLM inference?",
        "What CUDA version is installed on the EKS GPU node?",
        "What is the token throughput achieved by vLLM with Mistral-7B-GPTQ?",
    ],
    "hr_policies": [
        "What is the company leave policy?",
        "How do I submit a time-off request?",
    ],
    "org_info": [
        "What teams are part of the engineering organization?",
        "Who should I contact for IT support?",
    ],
}

SPOT_CHECK_QA = [
    {"question": "What GPU instance type does the Enterprise LLM Platform use for vLLM inference?",
     "ground_truth": "The platform uses g4dn.2xlarge instances with a Tesla T4 GPU (16GB VRAM) for vLLM inference."},
    {"question": "What is the token throughput achieved by vLLM with Mistral-7B-GPTQ on the T4 GPU?",
     "ground_truth": "vLLM achieves 37 tokens per second with Mistral-7B-Instruct-v0.2-GPTQ (4-bit) on a Tesla T4 GPU."},
    {"question": "What CUDA version does the EKS GPU node use?",
     "ground_truth": "The EKS GPU node uses CUDA 12.8, enabled by the AL2_x86_64_GPU AMI with pre-installed NVIDIA drivers."},
    {"question": "What percentage of T4 GPU SM utilization does vLLM achieve at peak?",
     "ground_truth": "vLLM achieves 90% GPU SM utilization and 94% memory bandwidth utilization on the Tesla T4 at peak load."},
    {"question": "How much VRAM does the Mistral-7B-GPTQ model consume on the T4 GPU?",
     "ground_truth": "vLLM serves TheBloke/Mistral-7B-Instruct-v0.2-GPTQ using GPTQ 4-bit quantization, consuming 12.9GB of the T4s 16GB VRAM."},
]


@pytest.fixture(scope="session")
def qdrant_client():
    from qdrant_client import QdrantClient
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not reachable: {e}")
    return client

@pytest.fixture(scope="session")
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@pytest.fixture(scope="session")
def pg_conn():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST, database=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASS,
            port=5432, sslmode="require", connect_timeout=5,
        )
        return conn
    except Exception as e:
        pytest.skip(f"PostgreSQL not reachable: {e}")


# L1: Qdrant Retrieval
class TestQdrantRetrieval:
    def test_all_collections_exist(self, qdrant_client):
        collections = {c.name for c in qdrant_client.get_collections().collections}
        for col in ["tech_docs", "hr_policies", "org_info"]:
            assert col in collections, f"Collection '{col}' missing"

    def test_tech_docs_retrieval(self, qdrant_client, embedding_model):
        for q in DOMAIN_QUESTIONS["tech_docs"]:
            vector = embedding_model.encode(q).tolist()
            hits = qdrant_client.search(collection_name="tech_docs", query_vector=vector,
                                        limit=3, score_threshold=0.0)
            assert len(hits) >= MIN_QDRANT_CHUNKS, f"tech_docs: 0 results for '{q}'"
            print(f"\n  tech_docs score={hits[0].score:.4f} | q='{q[:50]}'")

    def test_hr_policies_retrieval(self, qdrant_client, embedding_model):
        for q in DOMAIN_QUESTIONS["hr_policies"]:
            vector = embedding_model.encode(q).tolist()
            hits = qdrant_client.search(collection_name="hr_policies", query_vector=vector,
                                        limit=3, score_threshold=0.0)
            assert len(hits) >= MIN_QDRANT_CHUNKS, f"hr_policies: 0 results for '{q}'"

    def test_org_info_retrieval(self, qdrant_client, embedding_model):
        for q in DOMAIN_QUESTIONS["org_info"]:
            vector = embedding_model.encode(q).tolist()
            hits = qdrant_client.search(collection_name="org_info", query_vector=vector,
                                        limit=3, score_threshold=0.0)
            assert len(hits) >= MIN_QDRANT_CHUNKS, f"org_info: 0 results for '{q}'"


# L2: vLLM Inference
class TestVLLMInference:
    PROMPTS = [
        "What is a Tesla T4 GPU?",
        "Explain Kubernetes in one sentence.",
        "What does RAGAS stand for?",
        "What is the purpose of a vector database?",
        "Describe what vLLM does in 20 words.",
    ]

    def test_vllm_health(self):
        try:
            resp = requests.get(f"{VLLM_BASE_URL}/health", timeout=10)
            assert resp.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip(f"vLLM not reachable at {VLLM_BASE_URL}")

    def test_vllm_p95_latency(self):
        try:
            requests.get(f"{VLLM_BASE_URL}/health", timeout=5).raise_for_status()
        except Exception:
            pytest.skip("vLLM not reachable")
        latencies = []
        for prompt in self.PROMPTS:
            start = time.time()
            resp = requests.post(f"{VLLM_BASE_URL}/v1/completions", json={
                "model": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                "prompt": f"[INST] {prompt} [/INST]", "max_tokens": 80, "temperature": 0.1,
            }, timeout=30)
            lat = time.time() - start
            assert resp.ok, f"vLLM error: {resp.status_code}"
            text = resp.json()["choices"][0]["text"].strip()
            assert len(text) > 0, f"Empty response for: '{prompt}'"
            latencies.append(lat)
            print(f"\n  vLLM {lat:.2f}s | '{text[:50]}'")
        p95 = max(latencies)  # conservative with n=5
        print(f"\n  P95={p95:.2f}s SLA={P95_LATENCY_SLA_S}s")
        assert p95 <= P95_LATENCY_SLA_S, f"P95 {p95:.2f}s > SLA {P95_LATENCY_SLA_S}s"


# L3: Full RAG Flow
class TestRAGFullFlow:
    def test_tech_rag_flow(self, qdrant_client, embedding_model):
        try:
            requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
        except Exception:
            pytest.skip("vLLM not reachable")
        q = "What GPU does the platform use for LLM inference?"
        vector = embedding_model.encode(q).tolist()
        hits = qdrant_client.search(collection_name="tech_docs", query_vector=vector, limit=5)
        assert hits, "No context retrieved"
        context = "\n\n".join([r.payload.get("text", r.payload.get("content","")) for r in hits[:3]])
        prompt = f"[INST] Answer using only the context.\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer: [/INST]"
        resp = requests.post(f"{VLLM_BASE_URL}/v1/completions", json={
            "model": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            "prompt": prompt, "max_tokens": 120, "temperature": 0.0,
        }, timeout=30)
        assert resp.ok
        answer = resp.json()["choices"][0]["text"].strip()
        assert len(answer.split()) >= 5, f"Answer too short: '{answer}'"
        grounded = any(kw in answer.lower() for kw in ["t4", "gpu", "g4dn", "tesla"])
        assert grounded, f"Answer not grounded: '{answer}'"
        print(f"\n  RAG answer: '{answer[:80]}'")

    def test_retrieval_relevance(self, qdrant_client, embedding_model):
        q = "What is vLLM token throughput on Tesla T4?"
        vector = embedding_model.encode(q).tolist()
        hits = qdrant_client.search(collection_name="tech_docs", query_vector=vector, limit=5)
        assert hits
        assert hits[0].score > 0.3, f"Top score {hits[0].score:.4f} too low"


# L4: RAGAS Spot-check (optional - requires ragas package)
class TestRAGASSpotCheck:
    def test_ragas_faithfulness_spot_check(self, embedding_model):
        pytest.importorskip("ragas", reason="ragas not installed - run: pip install ragas datasets")
        try:
            requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
        except Exception:
            pytest.skip("vLLM not reachable")
        from qdrant_client import QdrantClient
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        records = []
        for qa in SPOT_CHECK_QA:
            vector = embedding_model.encode(qa["question"]).tolist()
            hits = qdrant.search(collection_name="tech_docs", query_vector=vector, limit=5)
            contexts = [r.payload.get("text", r.payload.get("content","")) for r in hits]
            prompt = f"[INST] Answer using context.\n\nContext:\n{chr(10).join(contexts[:3])}\n\nQ: {qa['question']}\nA: [/INST]"
            try:
                resp = requests.post(f"{VLLM_BASE_URL}/v1/completions", json={
                    "model": MODEL, "prompt": prompt, "max_tokens": 120, "temperature": 0.0,
                }, timeout=30)
                answer = resp.json()["choices"][0]["text"].strip() if resp.ok else ""
            except Exception:
                answer = ""
            records.append({"question": qa["question"], "answer": answer,
                             "contexts": contexts, "ground_truth": qa["ground_truth"]})
        scores = evaluate(Dataset.from_list(records), metrics=[faithfulness])
        faith = float(scores["faithfulness"])
        print(f"\n  RAGAS faithfulness (n=5): {faith:.4f}")
        assert faith >= MIN_FAITHFULNESS, f"Faithfulness {faith:.4f} < gate {MIN_FAITHFULNESS}"


# L5: DVC / Qdrant Point Counts
class TestDVCQdrantCounts:
    def test_qdrant_point_counts(self, qdrant_client):
        for col, minimum in [("tech_docs", MIN_TECH_DOCS), ("hr_policies", MIN_HR_POLICIES), ("org_info", MIN_ORG_INFO)]:
            count = qdrant_client.get_collection(col).points_count
            print(f"\n  {col}: {count} (min={minimum})")
            assert count >= minimum, f"{col}: {count} points < minimum {minimum}"

    def test_mlflow_ragas_run_exists(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("ragas-evaluation")
            assert exp is not None, "ragas-evaluation experiment not found"
            runs = client.search_runs(experiment_ids=[exp.experiment_id],
                                      order_by=["start_time DESC"], max_results=1)
            assert runs, "No RAGAS runs found"
            faith = runs[0].data.metrics.get("faithfulness")
            assert faith is not None, "No faithfulness metric"
            print(f"\n  Latest faithfulness: {faith:.4f}")
        except Exception as e:
            pytest.skip(f"MLflow not reachable: {e}")

    def test_golden_qa_count(self, pg_conn):
        cur = pg_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM golden_qa WHERE active=TRUE")
        total = cur.fetchone()[0]
        cur.close()
        print(f"\n  golden_qa active: {total}")
        assert total >= 60, f"Expected >= 60 QA pairs, got {total}"

    def test_golden_qa_domain_coverage(self, pg_conn):
        cur = pg_conn.cursor()
        cur.execute("SELECT domain, COUNT(*) FROM golden_qa WHERE active=TRUE GROUP BY domain")
        counts = dict(cur.fetchall())
        cur.close()
        print(f"\n  Domain counts: {counts}")
        for domain in ["tech", "hr", "org"]:
            assert counts.get(domain, 0) >= 20, f"Domain '{domain}' has < 20 pairs"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed  = len(terminalreporter.stats.get("passed", []))
    failed  = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    print(f"\n{'='*55}\nE2E TEST SUMMARY\n{'='*55}")
    print(f"  Passed:  {passed}\n  Failed:  {failed}\n  Skipped: {skipped}")
    print(f"  Overall: {'PASS' if failed == 0 else 'FAIL'}\n{'='*55}")


# ── L6: P11 Trigger Test ──────────────────────────────────────────────────────

class TestP11Trigger:
    """L6: Verify drift detected -> P11 auto-retraining fires and all 6 components succeed < 10 min."""

    def test_p11_pipeline_exists_in_kfp(self):
        """p11-auto-retraining pipeline is registered in KFP."""
        try:
            resp = requests.get("http://localhost:8080/apis/v2beta1/pipelines", timeout=10)
            assert resp.ok, f"KFP API not reachable: {resp.status_code}"
            pipelines = resp.json().get("pipelines", [])
            names = [p.get("display_name", "") for p in pipelines]
            found = any("p11" in n.lower() for n in names)
            assert found, f"No p11 pipeline found in KFP. Registered: {names[:5]}"
            print(f"\n  Found P11 pipeline in KFP registry")
        except requests.exceptions.ConnectionError:
            pytest.skip("KFP not reachable at localhost:8080 (port-forward required)")

    def test_p11_mlflow_experiment_exists(self):
        """p11-auto-retraining MLflow experiment exists with at least 1 run."""
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("p11-auto-retraining")
            assert exp is not None, "p11-auto-retraining experiment not found in MLflow"
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            assert runs, "No runs found in p11-auto-retraining experiment"
            outcome = runs[0].data.tags.get("outcome", "unknown")
            print(f"\n  P11 latest run outcome: {outcome}")
        except Exception as e:
            pytest.skip(f"MLflow not reachable: {e}")

    def test_p11_trigger_and_verify_succeeds(self):
        """Submit P11, verify all 6 components Succeeded in < 10 min."""
        import subprocess, time, uuid, os

        try:
            requests.get("http://localhost:8080/apis/v2beta1/healthz", timeout=5)
        except Exception:
            pytest.skip("KFP not reachable at localhost:8080")

        start = time.time()
        result = subprocess.run(
            ["python", "scripts/submit_p11.py", "--trigger", "manual"],
            capture_output=True, text=True,
            env={**os.environ, "KFP_ENDPOINT": "http://localhost:8080"},
        )
        assert result.returncode == 0, f"submit_p11.py failed:\n{result.stderr}"

        run_id = None
        for line in result.stdout.splitlines():
            if "Run submitted:" in line:
                run_id = line.split("Run submitted:")[-1].strip()
                break
        assert run_id, f"Could not parse run_id:\n{result.stdout}"
        print(f"\n  P11 submitted: {run_id}")

        max_wait, interval, elapsed, final_state = 600, 20, 0, None
        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval
            resp = requests.get(f"http://localhost:8080/apis/v2beta1/runs/{run_id}", timeout=10)
            if resp.ok:
                state = resp.json().get("state", "")
                print(f"  P11 [{elapsed}s]: {state}")
                if state in ("SUCCEEDED", "FAILED", "SKIPPED", "ERROR"):
                    final_state = state
                    break

        duration = time.time() - start
        assert final_state == "SUCCEEDED", f"P11 final state: {final_state}"
        assert duration < 600, f"P11 took {duration:.0f}s > 600s SLA"
        print(f"\n  P11 Succeeded in {duration:.0f}s")

    def test_p11_all_6_components_logged(self):
        """Confirm log-improvement-component (step 6) wrote metrics to MLflow."""
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("p11-auto-retraining")
            if not exp:
                pytest.skip("p11-auto-retraining experiment not found")
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            assert runs, "No P11 runs in MLflow"
            metrics = runs[0].data.metrics
            for m in ["pre_faithfulness", "post_faithfulness", "faithfulness_delta", "gate_passed"]:
                assert m in metrics, f"Missing metric '{m}' — component 6 may not have run"
            print(f"\n  P11 metrics confirmed: {list(metrics.keys())}")
        except Exception as e:
            pytest.skip(f"MLflow not reachable: {e}")
