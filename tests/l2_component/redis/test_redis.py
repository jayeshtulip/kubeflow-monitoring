"""L2 Component tests — Redis session store and rate limiter."""
import pytest
import time


@pytest.mark.l2
@pytest.mark.timeout(10)
def test_redis_set_get_delete(redis_client):
    """Basic set/get/delete round-trip."""
    redis_client.set("l2_test_key", "test_value", ex=60)
    val = redis_client.get("l2_test_key")
    assert val == "test_value"
    redis_client.delete("l2_test_key")
    assert redis_client.get("l2_test_key") is None


@pytest.mark.l2
@pytest.mark.timeout(10)
def test_redis_session_store_roundtrip(env_config):
    """Session save and load round-trip."""
    from src.storage.redis.session_store import save_session, load_session, delete_session
    state = {"query": "test", "workflow": "simple", "step": 1}
    ok = save_session("test-session-l2", state, ttl_seconds=60)
    assert ok, "save_session returned False"
    loaded = load_session("test-session-l2")
    assert loaded is not None, "load_session returned None"
    assert loaded["workflow"] == "simple"
    delete_session("test-session-l2")
    assert load_session("test-session-l2") is None


@pytest.mark.l2
@pytest.mark.timeout(10)
def test_redis_rate_limiter_allows_within_limit(env_config):
    """Rate limiter allows requests within limit."""
    from src.storage.redis.session_store import check_rate_limit
    client_id = "test-client-l2"
    allowed, count = check_rate_limit(client_id, max_requests=50, window_seconds=60)
    assert allowed is True, f"Rate limiter blocked valid request (count={count})"


@pytest.mark.l2
@pytest.mark.timeout(15)
def test_redis_query_cache_roundtrip(env_config):
    """Query response cache set and get."""
    from src.storage.redis.session_store import cache_response, get_cached_response
    query = "test query for cache l2"
    response = "This is a cached response"
    cache_response(query, response, ttl_seconds=60)
    cached = get_cached_response(query)
    assert cached == response, f"Cache returned {cached!r} instead of {response!r}"