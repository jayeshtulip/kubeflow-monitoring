"""Health and readiness endpoints."""
from __future__ import annotations
from fastapi import APIRouter
from src.api.schemas.query import HealthResponse
from src.serving.ollama.client import OllamaClient
from src.serving.vllm.client import VLLMClient
from pipelines.components.shared.base import EnvConfig, get_qdrant_client

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Kubernetes liveness probe."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    """Kubernetes readiness probe — checks downstream dependencies."""
    cfg = EnvConfig()

    # Check Ollama
    try:
        import httpx
        r = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    # Check vLLM
    vllm_ok = VLLMClient(cfg).health_check(timeout=3.0)

    # Check Qdrant
    try:
        qclient = get_qdrant_client(cfg)
        qclient.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    all_ok = ollama_ok and qdrant_ok
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        ollama_available=ollama_ok,
        vllm_available=vllm_ok,
        qdrant_available=qdrant_ok,
    )