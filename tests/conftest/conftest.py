"""
Shared pytest fixtures for L1-L4 test suites.
Loaded automatically by pytest via conftest.py discovery.
"""
from __future__ import annotations
import os
import pytest
from pipelines.components.shared.base import EnvConfig


@pytest.fixture(scope="session")
def env_config() -> EnvConfig:
    """EnvConfig populated from environment variables."""
    return EnvConfig()


@pytest.fixture(scope="session")
def qdrant_client(env_config: EnvConfig):
    """Qdrant client for component tests."""
    from qdrant_client import QdrantClient
    return QdrantClient(
        host=env_config.qdrant_host,
        port=env_config.qdrant_port,
        timeout=10,
    )


@pytest.fixture(scope="session")
def redis_client(env_config: EnvConfig):
    """Redis client for component tests."""
    import redis
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=True,
        socket_connect_timeout=5,
    )


@pytest.fixture(scope="session")
def postgres_conn(env_config: EnvConfig):
    """PostgreSQL connection for component tests."""
    import psycopg2
    conn = psycopg2.connect(
        host=env_config.postgres_host,
        dbname=env_config.postgres_db,
        user=env_config.postgres_user,
        password=env_config.postgres_password,
        connect_timeout=5,
    )
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def ollama_client(env_config: EnvConfig):
    """Ollama client for component tests."""
    from src.serving.ollama.client import OllamaClient
    return OllamaClient(env_config)


@pytest.fixture(scope="session")
def vllm_client(env_config: EnvConfig):
    """vLLM client for component tests."""
    from src.serving.vllm.client import VLLMClient
    return VLLMClient(env_config)


@pytest.fixture(scope="session")
def mlflow_client(env_config: EnvConfig):
    """MLflow client for component + L3 tests."""
    import mlflow
    mlflow.set_tracking_uri(env_config.mlflow_tracking_uri)
    return mlflow.tracking.MlflowClient()