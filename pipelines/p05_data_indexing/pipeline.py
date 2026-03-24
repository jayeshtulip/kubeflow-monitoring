"""
Kubeflow Pipeline 05 — Data Indexing
=====================================
Triggered by:
  - New Confluence page created / updated
  - New Jira ticket with label 'runbook'
  - Manual document upload via the FastAPI /upload endpoint
  - On-demand via Kubeflow UI

Steps:
  1. validate_document   — Great Expectations doc_ingestion_suite
  2. dvc_commit          — Version the raw document in DVC (S3 remote)
  3. chunk_and_embed     — Split text into 150-word chunks, generate embeddings
  4. upsert_to_qdrant    — Store vectors in the target collection
  5. validate_retrieval  — Test query to confirm document is retrievable
  6. log_lineage         — Record DVC SHA + Qdrant point count in MLflow + MLMD

SLA: < 10 seconds from document upload to queryable in Qdrant
     (validated by test_p05_data_indexing_pipeline in L3 suite)
"""


import kfp
from kfp import dsl
from kfp.dsl import (
    Output, Input, Artifact, Dataset, Metrics,
    component, pipeline, If, PipelineTaskFinalStatus,
)

from pipelines.components.great_expectations.component import (
    validate_document_component,
)
from pipelines.components.dvc.component import dvc_add_and_push_component


# ── Inline components (no external import — self-contained for KFP registry) ─

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "qdrant-client==1.8.0",
        "sentence-transformers==2.7.0",
    ],
)
def chunk_embed_upsert_component(
    doc_text: str,
    doc_id: str,
    source: str,
    collection: str,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    qdrant_host: str,
    qdrant_port: int,
    # Outputs
    indexing_metrics: Output[Metrics],
    indexing_report: Output[Artifact],
) -> int:
    """
    Chunk document → generate embeddings → upsert to Qdrant.
    Returns the number of chunks indexed.
    """
    import hashlib
    import json
    import time
    import uuid

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    # ── Chunk ─────────────────────────────────────────────────────────────────
    words = doc_text.split()
    if len(words) < 10:
        raise ValueError(f"Document too short to chunk: {len(words)} words")

    step = max(1, chunk_size - overlap)
    chunks, texts = [], []
    for i, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < 10 and i > 0:
            break
        chunks.append((i, start, end, " ".join(chunk_words)))
        texts.append(" ".join(chunk_words))

    # ── Embed ─────────────────────────────────────────────────────────────────
    t_embed = time.perf_counter()
    model = SentenceTransformer(embedding_model)
    vectors = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    embed_seconds = round(time.perf_counter() - t_embed, 3)

    # ── Upsert ────────────────────────────────────────────────────────────────
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )

    points = []
    for (idx, start, end, text), vector in zip(chunks, vectors):
        point_id = str(uuid.UUID(
            hashlib.md5(f"{doc_id}::{idx}".encode()).hexdigest()
        ))
        points.append(PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload={
                "text":        text,
                "doc_id":      doc_id,
                "source":      source,
                "collection":  collection,
                "chunk_index": idx,
                "start_word":  start,
                "end_word":    end,
            },
        ))

    t_upsert = time.perf_counter()
    client.upsert(collection_name=collection, points=points)
    upsert_seconds = round(time.perf_counter() - t_upsert, 3)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_dict = {
        "chunk_count":     len(chunks),
        "word_count":      len(words),
        "embed_seconds":   embed_seconds,
        "upsert_seconds":  upsert_seconds,
    }
    for k, v in metrics_dict.items():
        indexing_metrics.log_metric(k, v)

    report = {"doc_id": doc_id, "collection": collection, **metrics_dict}
    with open(indexing_report.path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Indexed {len(chunks)} chunks from '{doc_id}' into '{collection}'")
    return len(chunks)


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "qdrant-client==1.8.0",
        "sentence-transformers==2.7.0",
    ],
)
def validate_retrieval_component(
    doc_id: str,
    collection: str,
    test_query: str,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
    min_cosine_score: float,
    retrieval_validation_report: Output[Artifact],
) -> bool:
    """
    Run a test query against Qdrant and verify the indexed document is
    retrievable with cosine similarity >= min_cosine_score.

    Returns True if validation passes.
    """
    import json
    import time

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    vector = model.encode(test_query, normalize_embeddings=True).tolist()

    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
    t0 = time.perf_counter()

    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        with_payload=True,
        score_threshold=0.0,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    passed = len(hits) > 0 and hits[0].score >= min_cosine_score
    top_score = hits[0].score if hits else 0.0

    report = {
        "doc_id":          doc_id,
        "collection":      collection,
        "test_query":      test_query,
        "hits_found":      len(hits),
        "top_score":       round(top_score, 4),
        "min_required":    min_cosine_score,
        "latency_ms":      latency_ms,
        "validation_passed": passed,
    }
    with open(retrieval_validation_report.path, "w") as f:
        json.dump(report, f, indent=2)

    status = "PASS" if passed else "FAIL"
    print(f"Retrieval validation [{status}]: top_score={top_score:.3f}, latency={latency_ms}ms")
    return passed


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.0", "boto3==1.34.0"],
)
def log_indexing_lineage_component(
    doc_id: str,
    source: str,
    collection: str,
    dvc_sha_artifact: Input[Artifact],
    indexing_report: Input[Artifact],
    retrieval_validation_report: Input[Artifact],
    mlflow_tracking_uri: str,
) -> str:
    """
    Log complete lineage to MLflow:
      - DVC SHA of the raw document
      - Qdrant chunk count and collection
      - Retrieval validation result

    Returns the MLflow run_id for downstream MLMD logging.
    """
    import json
    import mlflow

    with open(dvc_sha_artifact.path) as f:
        dvc_data = json.load(f)
    with open(indexing_report.path) as f:
        index_data = json.load(f)
    with open(retrieval_validation_report.path) as f:
        retrieval_data = json.load(f)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("data-indexing")

    with mlflow.start_run(run_name=f"index-{doc_id}") as run:
        mlflow.log_params({
            "doc_id":     doc_id,
            "source":     source,
            "collection": collection,
            "dvc_sha":    dvc_data.get("dvc_sha", "unknown"),
        })
        mlflow.log_metrics({
            "chunk_count":          index_data.get("chunk_count", 0),
            "word_count":           index_data.get("word_count", 0),
            "embed_seconds":        index_data.get("embed_seconds", 0),
            "upsert_seconds":       index_data.get("upsert_seconds", 0),
            "retrieval_top_score":  retrieval_data.get("top_score", 0),
            "retrieval_latency_ms": retrieval_data.get("latency_ms", 0),
        })
        mlflow.set_tags({
            "retrieval_passed": str(retrieval_data.get("validation_passed", False)),
            "pipeline":         "P05-DataIndexing",
        })
        run_id = run.info.run_id

    print(f"Lineage logged to MLflow run: {run_id}")
    return run_id


@component(
    base_image="python:3.11-slim",
    packages_to_install=["requests==2.32.0"],
)
def reject_document_component(
    doc_id: str,
    source: str,
    validation_report: Input[Artifact],
    slack_webhook_url: str,
) -> None:
    """Alert on document rejection — runs when GX validation fails."""
    import json
    import requests

    with open(validation_report.path) as f:
        report = json.load(f)

    failed_checks = [c for c in report.get("checks", []) if not c["passed"]]
    msg = (
        f":warning: *P05 Document Rejected*\n"
        f"doc_id: `{doc_id}` | source: `{source}`\n"
        f"Failed checks: {', '.join(c['name'] for c in failed_checks)}\n"
        f"Details: {[c['detail'] for c in failed_checks]}"
    )
    print(msg)
    if slack_webhook_url and slack_webhook_url.startswith("https://"):
        try:
            requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

@pipeline(
    name="p05-data-indexing",
    description=(
        "Validate, version (DVC), chunk, embed, and upsert a document into Qdrant. "
        "SLA: < 10s from trigger to queryable. "
        "Triggered by: document upload, Confluence webhook, Jira webhook."
    ),
)
def data_indexing_pipeline(
    # Document inputs
    doc_text: str,
    doc_id: str,
    source: str = "confluence",
    collection: str = "tech_docs",

    # Chunking (Katib-tuned defaults)
    chunk_size: int = 150,
    overlap: int = 30,

    # Infrastructure
    qdrant_host: str = "qdrant-service.llm-platform-prod.svc.cluster.local",
    qdrant_port: int = 6333,
    embedding_model: str = "all-MiniLM-L6-v2",

    # DVC
    dvc_remote: str = "s3://llm-platform-dvc-remote",
    aws_region: str = "us-east-1",

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",

    # Validation
    min_cosine_score: float = 0.30,
    test_query: str = "",           # auto-derived from doc_id if empty

    # Alerting
    slack_webhook_url: str = "",
) -> None:

    # ── Step 1: Validate document (Great Expectations) ───────────────────────
    validation_task = validate_document_component(
        doc_text=doc_text,
        doc_id=doc_id,
        source=source,
    )
    validation_task.set_display_name("1 · Validate document (GX)")

    # ── Branch: if validation FAILED → reject and stop ───────────────────────
    with dsl.If(validation_task.outputs['Output'] == False, name="validation-failed"):
        reject_task = reject_document_component(
            doc_id=doc_id,
            source=source,
            validation_report=validation_task.outputs["validation_report"],
            slack_webhook_url=slack_webhook_url,
        )
        reject_task.set_display_name("1a · Reject document + alert")

    # ── Branch: if validation PASSED → continue pipeline ─────────────────────
    with dsl.If(validation_task.outputs['Output'] == True, name="validation-passed"):

        # ── Step 2: DVC — version the raw document ───────────────────────────
        dvc_task = dvc_add_and_push_component(
            local_path=f"/tmp/docs/{doc_id}.txt",
            dvc_remote=dvc_remote,
            aws_region=aws_region,
        )
        dvc_task.set_display_name("2 · DVC commit → S3")

        # ── Step 3: Chunk + Embed + Upsert to Qdrant ─────────────────────────
        index_task = chunk_embed_upsert_component(
            doc_text=doc_text,
            doc_id=doc_id,
            source=source,
            collection=collection,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model=embedding_model,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )
        index_task.set_display_name("3 · Chunk → Embed → Qdrant upsert")
        index_task.after(dvc_task)

        # ── Step 4: Validate retrieval ────────────────────────────────────────
        effective_query = test_query if test_query else doc_id
        retrieval_task = validate_retrieval_component(
            doc_id=doc_id,
            collection=collection,
            test_query=effective_query,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            min_cosine_score=min_cosine_score,
        )
        retrieval_task.set_display_name("4 · Validate retrieval (test query)")
        retrieval_task.after(index_task)

        # ── Step 5: Log lineage to MLflow ─────────────────────────────────────
        lineage_task = log_indexing_lineage_component(
            doc_id=doc_id,
            source=source,
            collection=collection,
            dvc_sha_artifact=dvc_task.outputs["dvc_sha_artifact"],
            indexing_report=index_task.outputs["indexing_report"],
            retrieval_validation_report=retrieval_task.outputs[
                "retrieval_validation_report"
            ],
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        lineage_task.set_display_name("5 · Log lineage → MLflow")
        lineage_task.after(retrieval_task)


# ── Compile pipeline ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=data_indexing_pipeline,
        package_path="pipelines/p05_data_indexing/pipeline.yaml",
    )
    print("Compiled → pipelines/p05_data_indexing/pipeline.yaml")

