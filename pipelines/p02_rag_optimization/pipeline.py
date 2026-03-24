"""
Pipeline 02 - RAG Optimization
Tests 9 chunking strategies: 50/100/150 words x 10/20/30 overlap.
Measures retrieval accuracy per strategy.
Logs to MLflow experiment: rag-optimization.
Output: optimal chunk_size + overlap.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Metrics, component, pipeline


@component(
    base_image="python:3.11-slim",
    packages_to_install=["qdrant-client==1.8.0", "sentence-transformers==2.7.0",
                          "psycopg2-binary==2.9.9"],
)
def test_chunking_strategy_component(
    chunk_size: int,
    overlap: int,
    top_k: int,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    strategy_result: Output[Artifact],
) -> float:
    """Test one chunking strategy. Returns retrieval accuracy 0-1."""
    import json, psycopg2, psycopg2.extras
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    conn = psycopg2.connect(host=postgres_host, dbname=postgres_db,
                            user=postgres_user, password=postgres_password)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            "SELECT question, source_doc_id FROM golden_qa"
            " WHERE active=TRUE AND source_doc_id IS NOT NULL LIMIT 30")
        pairs = cur.fetchall()
    conn.close()
    if not pairs:
        result = {"chunk_size": chunk_size, "overlap": overlap, "accuracy": 0.0}
        with open(strategy_result.path, "w") as f: json.dump(result, f)
        return 0.0
    model   = SentenceTransformer(embedding_model)
    qclient = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
    correct = 0
    for pair in pairs:
        vec = model.encode(pair["question"], normalize_embeddings=True).tolist()
        try:
            hits = qclient.search("tech_docs", query_vector=vec,
                                  limit=top_k, with_payload=True)
            if hits and any(h.payload.get("doc_id","") == pair["source_doc_id"]
                           for h in hits):
                correct += 1
        except Exception:
            pass
    accuracy = correct / len(pairs)
    result = {"chunk_size": chunk_size, "overlap": overlap, "top_k": top_k,
              "accuracy": round(accuracy, 4), "tested": len(pairs)}
    with open(strategy_result.path, "w") as f: json.dump(result, f, indent=2)
    print(f"chunk={chunk_size} overlap={overlap} accuracy={accuracy:.3f}")
    return accuracy


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2", "boto3==1.34.69"],
)
def select_best_strategy_component(
    mlflow_tracking_uri: str,
    best_params_artifact: Output[Artifact],
) -> str:
    """Query MLflow for best chunking strategy across all runs."""
    import json, mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_names=["rag-optimization"],
        order_by=["metrics.accuracy DESC"],
        max_results=1)
    if runs:
        best = {
            "best_chunk_size": int(runs[0].data.params.get("chunk_size", 150)),
            "best_overlap":    int(runs[0].data.params.get("overlap", 30)),
            "best_accuracy":   runs[0].data.metrics.get("accuracy", 0),
        }
    else:
        best = {"best_chunk_size": 150, "best_overlap": 30, "best_accuracy": 0}
    mlflow.set_experiment("rag-optimization")
    with mlflow.start_run(run_name="rag-best-selection"):
        mlflow.log_params(best)
        mlflow.set_tag("pipeline", "P02-RAGOptimization")
    with open(best_params_artifact.path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best: {best}")
    return json.dumps(best)


@pipeline(
    name="p02-rag-optimization",
    description="Test 9 chunking strategies. Output optimal chunk_size + overlap.",
)
def rag_optimization_pipeline(
    qdrant_host: str = "qdrant-service.llm-platform-prod.svc.cluster.local",
    qdrant_port: int = 6333,
    embedding_model: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    postgres_host: str = "llm-platform.xxxxxxxx.rds.amazonaws.com",
    postgres_db: str = "llm_platform",
    postgres_user: str = "llm_admin",
    postgres_password: str = "",
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
) -> None:
    tasks = []
    for chunk_size in [50, 100, 150]:
        for overlap in [10, 20, 30]:
            t = test_chunking_strategy_component(
                chunk_size=chunk_size, overlap=overlap, top_k=top_k,
                qdrant_host=qdrant_host, qdrant_port=qdrant_port,
                embedding_model=embedding_model,
                postgres_host=postgres_host, postgres_db=postgres_db,
                postgres_user=postgres_user, postgres_password=postgres_password)
            t.set_display_name(f"Test chunk={chunk_size} overlap={overlap}")
            tasks.append(t)
    select_task = select_best_strategy_component(
        mlflow_tracking_uri=mlflow_tracking_uri)
    select_task.set_display_name("Select best chunking strategy")
    for t in tasks:
        select_task.after(t)


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=rag_optimization_pipeline,
        package_path="pipelines/p02_rag_optimization/pipeline.yaml")
    print("Compiled P02")