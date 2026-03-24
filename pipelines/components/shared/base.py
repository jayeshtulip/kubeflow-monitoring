from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


@dataclass
class EnvConfig:
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_s3_artifact_root: str = field(
        default_factory=lambda: os.environ.get(
            "MLFLOW_S3_ARTIFACT_ROOT", "s3://llm-platform-mlflow-artifacts"
        )
    )
    qdrant_host: str = field(
        default_factory=lambda: os.environ.get("QDRANT_HOST", "localhost")
    )
    qdrant_port: int = field(
        default_factory=lambda: int(os.environ.get("QDRANT_PORT", "6333"))
    )
    postgres_host: str = field(
        default_factory=lambda: os.environ.get("POSTGRES_HOST", "localhost")
    )
    postgres_db: str = field(
        default_factory=lambda: os.environ.get("POSTGRES_DB", "llm_platform")
    )
    postgres_user: str = field(
        default_factory=lambda: os.environ.get("POSTGRES_USER", "llm_admin")
    )
    postgres_password: str = field(
        default_factory=lambda: os.environ.get("POSTGRES_PASSWORD", "")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://ollama-service:11434"
        )
    )
    vllm_api_base: str = field(
        default_factory=lambda: os.environ.get(
            "VLLM_API_BASE", "http://vllm-service:8000/v1"
        )
    )
    dvc_remote_bucket: str = field(
        default_factory=lambda: os.environ.get(
            "DVC_REMOTE_BUCKET", "s3://llm-platform-dvc-remote"
        )
    )
    aws_region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    kubeflow_host: str = field(
        default_factory=lambda: os.environ.get("KUBEFLOW_HOST", "http://localhost:8080")
    )
    embedding_dims: int = 384
    langsmith_api_key: str = field(
        default_factory=lambda: os.environ.get("LANGSMITH_API_KEY", "")
    )
    langsmith_project: str = field(
        default_factory=lambda: os.environ.get("LANGSMITH_PROJECT", "llm-platform-prod")
    )
    slack_webhook_url: str = field(
        default_factory=lambda: os.environ.get("SLACK_WEBHOOK_URL", "")
    )


def get_mlflow_client(cfg: EnvConfig) -> MlflowClient:
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)


def get_qdrant_client(cfg: EnvConfig) -> QdrantClient:
    return QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port, timeout=30)


def ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    dims: int = 384,
) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {collection_name}")
    else:
        print(f"Qdrant collection already exists: {collection_name}")


@dataclass
class StepResult:
    success: bool
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "metrics": self.metrics,
        }
