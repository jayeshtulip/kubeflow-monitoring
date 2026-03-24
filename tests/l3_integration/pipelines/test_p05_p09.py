"""
L3 Integration tests — Pipeline 05 (Data Indexing) and Pipeline 09 (RAGAS Evaluation)

Run with:
    pytest tests/l3_integration/pipelines/test_p05_p09.py -m l3 -v

These tests submit real Kubeflow pipeline runs against the staging cluster
and assert on outcomes. They are gated in the merge-gate CI workflow
(GitHub Actions: merge-gate.yml) and in the nightly regression suite.

Fixtures (from tests/conftest/conftest.py):
    kfp_client       — KFP SDK v2 client authenticated via OIDC
    qdrant_client    — Qdrant client pointing to staging collection
    mlflow_client    — MLflow tracking client
    postgres_conn    — psycopg2 connection to staging DB
    env_config       — EnvConfig populated from staging environment vars
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import mlflow
from kfp import client as kfp_sdk
from qdrant_client import QdrantClient

# ── Fixtures (declared here; full implementations in conftest.py) ─────────────

@pytest.fixture(scope="module")
def kfp_client(env_config):
    return kfp_sdk.Client(host=env_config.kubeflow_host)


@pytest.fixture(scope="module")
def qdrant(env_config):
    from qdrant_client import QdrantClient
    return QdrantClient(host=env_config.qdrant_host, port=env_config.qdrant_port)


@pytest.fixture(scope="module")
def mlflow_client(env_config):
    mlflow.set_tracking_uri(env_config.mlflow_tracking_uri)
    return mlflow.tracking.MlflowClient()


# ── Helper ────────────────────────────────────────────────────────────────────

def wait_for_run(
    kfp_client,
    run_id: str,
    timeout_seconds: int = 600,
    poll_interval: int = 10,
) -> dict:
    """Poll until a KFP run reaches a terminal state. Returns run detail."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = kfp_client.get_run(run_id=run_id)
        state = run.state
        if state in ("SUCCEEDED", "FAILED", "ERROR", "SKIPPED"):
            return {"state": state, "run": run}
        time.sleep(poll_interval)
    raise TimeoutError(f"Run {run_id} did not complete within {timeout_seconds}s")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 05 TESTS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_DOC = {
    "doc_text": (
        "The EKS payment service experienced intermittent timeouts due to "
        "pod evictions caused by memory pressure from the daily-report-generator "
        "CronJob. The CronJob started at 14:15, causing memory spike on three nodes. "
        "Pod evictions at 14:20 led to orphaned RDS connections and subsequent "
        "connection pool exhaustion (98/100 connections). Resolution: increase "
        "pod memory limits to 2Gi and add PodDisruptionBudget. "
        "Long-term fix: schedule CronJob during off-peak hours (02:00 UTC). "
        "Monitoring: add CloudWatch alarm on DatabaseConnections > 80. "
        "This incident affected the payment namespace for approximately 8 minutes. "
        "MTTR was reduced from 45 minutes to 8 minutes after implementing the "
        "automated runbook with LangGraph Planner-Executor-Critic workflow."
    ),
    "doc_id": "JIRA-TEST-001",
    "source": "jira",
    "collection": "tech_docs",
}


@pytest.mark.l3
@pytest.mark.timeout(120)
def test_p05_pipeline_runs_successfully(kfp_client, env_config):
    """P05 pipeline completes with SUCCEEDED state for a valid document."""
    from pipelines.p05_data_indexing.pipeline import data_indexing_pipeline

    run = kfp_client.create_run_from_pipeline_func(
        data_indexing_pipeline,
        arguments={
            **SAMPLE_DOC,
            "qdrant_host":          env_config.qdrant_host,
            "qdrant_port":          env_config.qdrant_port,
            "mlflow_tracking_uri":  env_config.mlflow_tracking_uri,
            "dvc_remote":           env_config.dvc_remote_bucket,
            "aws_region":           env_config.aws_region,
        },
        run_name="test-p05-valid-doc",
        experiment_name="l3-integration-tests",
    )
    result = wait_for_run(kfp_client, run.run_id, timeout_seconds=90)
    assert result["state"] == "SUCCEEDED", (
        f"P05 pipeline FAILED — check Kubeflow UI for run {run.run_id}"
    )


@pytest.mark.l3
@pytest.mark.timeout(30)
def test_p05_document_retrievable_after_indexing(qdrant, env_config):
    """
    After P05 runs, the indexed document is retrievable from Qdrant
    with cosine similarity >= 0.30.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(env_config.embedding_model)
    query = "payment service timeout pod eviction"
    vector = model.encode(query, normalize_embeddings=True).tolist()

    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    hits = qdrant.search(
        collection_name="tech_docs",
        query_vector=vector,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value="JIRA-TEST-001"))]
        ),
        score_threshold=0.0,
    )

    assert len(hits) > 0, "No chunks found for JIRA-TEST-001 in tech_docs"
    assert hits[0].score >= 0.30, (
        f"Top chunk score {hits[0].score:.3f} < 0.30 — retrieval quality too low"
    )


@pytest.mark.l3
@pytest.mark.timeout(20)
def test_p05_lineage_logged_to_mlflow(mlflow_client):
    """P05 logs a run to MLflow experiment 'data-indexing' for the test doc."""
    runs = mlflow_client.search_runs(
        experiment_names=["data-indexing"],
        filter_string="params.doc_id = 'JIRA-TEST-001'",
        max_results=1,
    )
    assert len(runs) > 0, "No MLflow run found for doc_id=JIRA-TEST-001"
    run = runs[0]
    assert run.data.metrics.get("chunk_count", 0) > 0, "chunk_count metric not logged"
    assert run.data.params.get("dvc_sha"), "DVC SHA not recorded in MLflow params"


@pytest.mark.l3
@pytest.mark.timeout(30)
def test_p05_rejects_invalid_document(kfp_client, env_config):
    """P05 pipeline marks as SUCCEEDED but routes through rejection branch for invalid doc."""
    from pipelines.p05_data_indexing.pipeline import data_indexing_pipeline

    run = kfp_client.create_run_from_pipeline_func(
        data_indexing_pipeline,
        arguments={
            "doc_text":   "too short",    # < 50 words — should fail GX validation
            "doc_id":     "INVALID-001",
            "source":     "test",
            "collection": "tech_docs",
            "qdrant_host": env_config.qdrant_host,
            "qdrant_port": env_config.qdrant_port,
            "mlflow_tracking_uri": env_config.mlflow_tracking_uri,
        },
        run_name="test-p05-invalid-doc",
        experiment_name="l3-integration-tests",
    )
    result = wait_for_run(kfp_client, run.run_id, timeout_seconds=60)
    # Pipeline itself succeeds (rejection is a valid branch, not an error)
    assert result["state"] == "SUCCEEDED"

    # Verify document NOT indexed in Qdrant
    from sentence_transformers import SentenceTransformer
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    model = SentenceTransformer(env_config.embedding_model)
    vector = model.encode("invalid", normalize_embeddings=True).tolist()
    hits = qdrant.search(
        collection_name="tech_docs",
        query_vector=vector,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value="INVALID-001"))]
        ),
    )
    assert len(hits) == 0, "Invalid document was indexed despite failing GX validation"


@pytest.mark.l3
@pytest.mark.timeout(15)
def test_p05_indexing_within_sla(kfp_client):
    """Last P05 run completed within 10-second SLA (check MLflow metrics)."""
    runs = mlflow_client.search_runs(
        experiment_names=["data-indexing"],
        filter_string="params.doc_id = 'JIRA-TEST-001'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    assert len(runs) > 0
    total_seconds = runs[0].data.metrics.get("upsert_seconds", 999)
    assert total_seconds < 10.0, (
        f"P05 upsert took {total_seconds:.2f}s — exceeds 10s SLA"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE 09 TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.l3
@pytest.mark.timeout(1800)   # 30 min SLA for 200 samples
def test_p09_pipeline_runs_successfully(kfp_client, env_config):
    """P09 pipeline completes with SUCCEEDED state (full 200-sample eval)."""
    from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline

    run = kfp_client.create_run_from_pipeline_func(
        ragas_evaluation_pipeline,
        arguments={
            "qa_limit":              50,    # use 50 samples in CI (full 200 in nightly)
            "qdrant_host":           env_config.qdrant_host,
            "qdrant_port":           env_config.qdrant_port,
            "postgres_host":         env_config.postgres_host,
            "postgres_db":           env_config.postgres_db,
            "postgres_user":         env_config.postgres_user,
            "postgres_password":     env_config.postgres_password,
            "ollama_base_url":       env_config.ollama_base_url,
            "mlflow_tracking_uri":   env_config.mlflow_tracking_uri,
            "run_name":              "l3-test-run",
        },
        run_name="test-p09-ragas-l3",
        experiment_name="l3-integration-tests",
    )
    result = wait_for_run(kfp_client, run.run_id, timeout_seconds=1800)
    assert result["state"] == "SUCCEEDED", (
        f"P09 pipeline FAILED — check Kubeflow UI for run {run.run_id}"
    )


@pytest.mark.l3
@pytest.mark.timeout(30)
def test_p09_all_six_metrics_logged_to_mlflow(mlflow_client):
    """P09 logs all 6 RAGAS metrics to MLflow experiment 'ragas-evaluation'."""
    runs = mlflow_client.search_runs(
        experiment_names=["ragas-evaluation"],
        filter_string="tags.pipeline = 'P09-RAGAS'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    assert len(runs) > 0, "No P09 RAGAS run found in MLflow"

    metrics = runs[0].data.metrics
    expected_metrics = [
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "answer_correctness", "hallucination_rate",
    ]
    for m in expected_metrics:
        assert m in metrics, f"Metric '{m}' not logged to MLflow"
        assert 0.0 <= metrics[m] <= 1.0, (
            f"Metric '{m}' = {metrics[m]} out of expected [0, 1] range"
        )


@pytest.mark.l3
@pytest.mark.timeout(30)
def test_p09_faithfulness_above_hard_block(mlflow_client):
    """Most recent P09 run has faithfulness >= 0.85 (hard block threshold)."""
    runs = mlflow_client.search_runs(
        experiment_names=["ragas-evaluation"],
        filter_string="tags.pipeline = 'P09-RAGAS'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    assert len(runs) > 0
    faith = runs[0].data.metrics["faithfulness"]
    assert faith >= 0.85, (
        f"Faithfulness {faith:.3f} is below hard block 0.85 — "
        "P09 should have triggered P11 retraining"
    )


@pytest.mark.l3
@pytest.mark.timeout(30)
def test_p09_gate_tag_matches_faithfulness(mlflow_client):
    """MLflow gate_passed tag is consistent with the faithfulness score."""
    runs = mlflow_client.search_runs(
        experiment_names=["ragas-evaluation"],
        filter_string="tags.pipeline = 'P09-RAGAS'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    assert len(runs) > 0
    tags = runs[0].data.tags
    faith = runs[0].data.metrics["faithfulness"]
    halluc = runs[0].data.metrics["hallucination_rate"]

    gate_passed = tags.get("gate_passed", "False") == "True"
    expected_pass = faith >= 0.85 and halluc <= 0.15

    assert gate_passed == expected_pass, (
        f"gate_passed tag ({gate_passed}) inconsistent with scores: "
        f"faithfulness={faith:.3f}, hallucination={halluc:.3f}"
    )


@pytest.mark.l3
@pytest.mark.timeout(30)
def test_p09_promotion_or_block_artifact_exists(kfp_client):
    """P09 always produces either a promotion_result or block_artifact."""
    runs = mlflow_client.search_runs(
        experiment_names=["ragas-evaluation"],
        filter_string="tags.pipeline = 'P09-RAGAS'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    assert len(runs) > 0
    # Either promotion logged or gate_failures recorded
    tags = runs[0].data.tags
    assert "gate_passed" in tags, "gate_passed tag missing from P09 MLflow run"
    assert "gate_failures" in tags, "gate_failures tag missing from P09 MLflow run"
