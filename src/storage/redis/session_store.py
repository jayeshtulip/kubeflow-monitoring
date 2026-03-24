"""
Redis session store.
Responsibilities:
  - LangGraph session state (conversation context)
  - Query deduplication (5 min TTL)
  - Rate limit counters (50 req/min per client)
  - RAGAS evaluation cache
"""
from __future__ import annotations
import json
import time
import hashlib
from typing import Any
from redis import Redis
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

_CLIENT: Redis | None = None


def get_redis_client(cfg: EnvConfig | None = None) -> redis.Redis:
    global _CLIENT
    if _CLIENT is None:
        cfg = cfg or EnvConfig()
        _CLIENT = Redis(
            host=cfg.qdrant_host.replace("qdrant", "redis").replace("6333", ""),
            port=6379,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _CLIENT


# ── Session state ─────────────────────────────────────────────────────────

def save_session(session_id: str, state: dict[str, Any],
                 ttl_seconds: int = 3600) -> bool:
    """Save LangGraph session state. Returns True on success."""
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        client.setex(key, ttl_seconds, json.dumps(state))
        logger.debug("Session saved: %s", session_id)
        return True
    except Exception as exc:
        logger.error("Redis save_session error: %s", exc)
        return False


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load LangGraph session state. Returns None if not found."""
    try:
        client = get_redis_client()
        raw = client.get(f"session:{session_id}")
        if raw:
            return json.loads(raw)
        return None
    except Exception as exc:
        logger.error("Redis load_session error: %s", exc)
        return None


def delete_session(session_id: str) -> None:
    try:
        get_redis_client().delete(f"session:{session_id}")
    except Exception as exc:
        logger.error("Redis delete_session error: %s", exc)


# ── Rate limiting ─────────────────────────────────────────────────────────

def check_rate_limit(
    client_id: str,
    max_requests: int = 50,
    window_seconds: int = 60,
) -> tuple[bool, int]:
    """
    Sliding window rate limiter.
    Returns (allowed, current_count).
    """
    try:
        r = get_redis_client()
        key = f"ratelimit:{client_id}"
        pipe = r.pipeline()
        now = time.time()
        window_start = now - window_seconds
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds * 2)
        results = pipe.execute()
        count = results[2]
        allowed = count <= max_requests
        if not allowed:
            logger.warning("Rate limit exceeded for %s: %d/%d",
                           client_id, count, max_requests)
        return allowed, count
    except Exception as exc:
        logger.error("Redis rate_limit error: %s", exc)
        return True, 0  # Fail open


# ── Query deduplication ───────────────────────────────────────────────────

def get_cached_response(query: str, ttl_seconds: int = 300) -> str | None:
    """Return cached response for identical query within TTL window."""
    try:
        key = f"dedup:{hashlib.md5(query.encode()).hexdigest()}"
        return get_redis_client().get(key)
    except Exception:
        return None


def cache_response(query: str, response: str, ttl_seconds: int = 300) -> None:
    """Cache a response for query deduplication."""
    try:
        key = f"dedup:{hashlib.md5(query.encode()).hexdigest()}"
        get_redis_client().setex(key, ttl_seconds, response)
    except Exception as exc:
        logger.error("Redis cache_response error: %s", exc)


# ── RAGAS cache ───────────────────────────────────────────────────────────

def cache_ragas_scores(run_id: str, scores: dict[str, Any],
                       ttl_seconds: int = 86400) -> None:
    """Cache RAGAS scores for a run_id (24h default)."""
    try:
        key = f"ragas:{run_id}"
        get_redis_client().setex(key, ttl_seconds, json.dumps(scores))
    except Exception as exc:
        logger.error("Redis cache_ragas_scores error: %s", exc)


def get_ragas_scores(run_id: str) -> dict[str, Any] | None:
    try:
        raw = get_redis_client().get(f"ragas:{run_id}")
        return json.loads(raw) if raw else None
    except Exception:
        return None