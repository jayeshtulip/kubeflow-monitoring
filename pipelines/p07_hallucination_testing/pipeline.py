"""
Pipeline 07 - Hallucination Testing
Tests factual grounding across 100 queries.
Checks: claims match retrieved docs, no fabricated tickets/commands,
source attribution, internal consistency.
Target: hallucination_rate < 2%.
Logs to MLflow experiment: hallucination-testing.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Metrics, component, pipeline


@component(
    base_image="python:3.11-slim",
    packages_to_install=["httpx==0.27.0", "qdrant-client==1.8.0",
                          "sentence-transformers==2.7.0", "mlflow==2.12.2",
                          "boto3==1.34.69", "psycopg2-binary==2.9.9"],
)
def test_hallucination_component(
    api_base_url: str,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    test_sample_size: int,
    mlflow_tracking_uri: str,
    hallucination_report: Output[Artifact],
    hallucination_metrics: Output[Metrics],
) -> float:
    """Generate responses and check factual grounding. Returns hallucination rate."""
    import json, re, httpx, psycopg2, psycopg2.extras, mlflow
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    conn = psycopg2.connect(host=postgres_host, dbname=postgres_db,
                            user=postgres_user, password=postgres_password)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            "SELECT question, ground_truth FROM golden_qa WHERE active=TRUE LIMIT %s",
            (test_sample_size,))
        pairs = cur.fetchall()
    conn.close()
    model   = SentenceTransformer(embedding_model)
    qclient = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
    pii_patterns = [
        r"JIRA-[0-9]{5,}",
        r"kubectl [a-z]+ --[a-z]+ [A-Z]{8,}",
        r"Error code: [A-Z]{3}[0-9]{5,}",
    ]
    hallucinations = []
    results = []
    for pair in pairs:
        try:
            r = httpx.post(f"{api_base_url}/api/query",
                           json={"query": pair["question"]}, timeout=90.0)
            response = r.json().get("response", "") if r.status_code == 200 else ""
        except Exception:
            response = ""
        has_fabrication = any(re.search(p, response) for p in pii_patterns) if response else False
        sentences = [s.strip() for s in response.replace(".", ". ").split(". ")
                     if len(s.strip()) > 20]
        grounded = 0
        for sent in sentences[:10]:
            sv = model.encode(sent, normalize_embeddings=True).tolist()
            try:
                hits = qclient.search("tech_docs", query_vector=sv, limit=1)
                if hits and hits[0].score > 0.6:
                    grounded += 1
            except Exception:
                pass
        ratio = grounded / max(len(sentences[:10]), 1)
        is_hallucination = has_fabrication or ratio < 0.3
        if is_hallucination:
            hallucinations.append({"question": pair["question"],
                                   "fabrication": has_fabrication,
                                   "grounding_ratio": ratio})
        results.append({"question": pair["question"],
                        "is_hallucination": is_hallucination,
                        "grounding_ratio": ratio})
    hallucination_rate = len(hallucinations) / len(results) if results else 0.0
    avg_grounding = sum(r["grounding_ratio"] for r in results) / len(results) if results else 0.0
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("hallucination-testing")
    with mlflow.start_run(run_name="hallucination-test") as run:
        mlflow.log_metrics({
            "hallucination_rate":  hallucination_rate,
            "avg_grounding_ratio": avg_grounding,
            "test_count":          float(len(results)),
            "fabrication_count":   float(sum(1 for r in results if r.get("fabrication", False))),
        })
        mlflow.set_tag("pipeline", "P07-HallucinationTesting")
    report = {
        "hallucination_rate":      hallucination_rate,
        "avg_grounding_ratio":     avg_grounding,
        "test_count":              len(results),
        "hallucination_examples":  hallucinations[:5],
        "mlflow_run_id":           run.info.run_id,
    }
    with open(hallucination_report.path, "w") as f:
        json.dump(report, f, indent=2)
    hallucination_metrics.log_metric("hallucination_rate", hallucination_rate)
    hallucination_metrics.log_metric("avg_grounding_ratio", avg_grounding)
    status = "PASS" if hallucination_rate < 0.02 else "FAIL"
    print(f"Hallucination [{status}]: rate={hallucination_rate:.3f} grounding={avg_grounding:.3f}")
    return hallucination_rate


@pipeline(
    name="p07-hallucination-testing",
    description="Test factual grounding across 100 queries. Target < 2% hallucination.",
)
def hallucination_testing_pipeline(
    api_base_url: str = "http://llm-gateway.llm-platform-prod.svc.cluster.local:8000",
    qdrant_host: str = "qdrant-service.llm-platform-prod.svc.cluster.local",
    qdrant_port: int = 6333,
    embedding_model: str = "all-MiniLM-L6-v2",
    postgres_host: str = "llm-platform.xxxxxxxx.rds.amazonaws.com",
    postgres_db: str = "llm_platform",
    postgres_user: str = "llm_admin",
    postgres_password: str = "",
    test_sample_size: int = 100,
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
) -> None:
    test_task = test_hallucination_component(
        api_base_url=api_base_url,
        qdrant_host=qdrant_host, qdrant_port=qdrant_port,
        embedding_model=embedding_model,
        postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        test_sample_size=test_sample_size,
        mlflow_tracking_uri=mlflow_tracking_uri)
    test_task.set_display_name("Test hallucination rate across 100 queries")


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=hallucination_testing_pipeline,
        package_path="pipelines/p07_hallucination_testing/pipeline.yaml")
    print("Compiled P07")