"""
Pytest root conftest — adds project root to sys.path and provides
shared fixtures for L1-L4 test suites.

Fixtures available to all tests:
    env_config    — EnvConfig loaded from environment variables
    qdrant_client — live Qdrant client (L2+ tests)
    redis_client  — live Redis client  (L2+ tests)
    postgres_conn — live PostgreSQL connection (L2+ tests)
    mlflow_client — MLflow tracking client (L2+ tests)
    kfp_client    — Kubeflow Pipelines client (L3 tests)
"""
import os
import sys
import pytest

# Ensure project root is on PYTHONPATH regardless of where pytest is invoked
sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Core fixture: EnvConfig
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def env_config():
    from pipelines.components.shared.base import EnvConfig
    return EnvConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Storage fixtures (skipped automatically if services not reachable)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_client(env_config):
    pytest.importorskip("qdrant_client", reason="qdrant_client not installed")
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(
            host=env_config.qdrant_host,
            port=env_config.qdrant_port,
            timeout=5,
        )
        client.get_collections()   # connectivity check
        return client
    except Exception as e:
        pytest.skip(f"Qdrant not reachable at {env_config.qdrant_host}:{env_config.qdrant_port} — {e}")


@pytest.fixture(scope="session")
def redis_client(env_config):
    pytest.importorskip("redis", reason="redis not installed")
    import redis as redis_lib
    try:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        client = redis_lib.Redis(host=host, port=port, decode_responses=True, socket_timeout=3)
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not reachable — {e}")


@pytest.fixture(scope="session")
def postgres_conn(env_config):
    pytest.importorskip("psycopg2", reason="psycopg2 not installed")
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=env_config.postgres_host,
            dbname=env_config.postgres_db,
            user=env_config.postgres_user,
            password=env_config.postgres_password,
            connect_timeout=5,
        )
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not reachable at {env_config.postgres_host} — {e}")


@pytest.fixture(scope="session")
def mlflow_client(env_config):
    pytest.importorskip("mlflow", reason="mlflow not installed")
    import mlflow
    mlflow.set_tracking_uri(env_config.mlflow_tracking_uri)
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def kfp_client(env_config):
    pytest.importorskip("kfp", reason="kfp not installed")
    try:
        from kfp import client as kfp_sdk
        return kfp_sdk.Client(host=env_config.kubeflow_host)
    except Exception as e:
        pytest.skip(f"Kubeflow not reachable at {env_config.kubeflow_host} — {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers so they don't generate warnings."""
    config.addinivalue_line("markers", "l1: Infrastructure tests (require AWS)")
    config.addinivalue_line("markers", "l2: Component tests (require live services)")
    config.addinivalue_line("markers", "l3: Integration tests (require full platform)")
    config.addinivalue_line("markers", "l4: Performance tests (require H100 vLLM)")
    config.addinivalue_line("markers", "aws: Tests requiring live AWS resources")
    config.addinivalue_line("markers", "live: Tests requiring live service endpoints")
    config.addinivalue_line("markers", "slow: Tests taking more than 30 seconds")

@pytest.fixture(scope="session")
def vllm_client(env_config):
    """VLLMClient pointing to vLLM serving endpoint. Skips if not reachable."""
    pytest.importorskip("httpx", reason="httpx not installed")
    import httpx
    base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8080/v1")
    try:
        resp = httpx.get(base_url.replace("/v1", "/health"), timeout=3)
        if resp.status_code != 200:
            pytest.skip(f"vLLM not healthy at {base_url} (status {resp.status_code})")
    except Exception as e:
        pytest.skip(f"vLLM not reachable at {base_url} — {e}")
    from src.serving.vllm.client import VLLMClient
    return VLLMClient(api_base=base_url)
