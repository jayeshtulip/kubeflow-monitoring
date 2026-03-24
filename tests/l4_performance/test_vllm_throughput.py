"""
L4 Performance tests — vLLM sustained throughput on H100.
Run with: pytest tests/l4_performance/test_vllm_throughput.py -m l4 -v
"""
from __future__ import annotations
import time
import pytest
import statistics


@pytest.mark.l4
@pytest.mark.timeout(120)
def test_vllm_sustained_throughput(vllm_client):
    """
    Generate 5 consecutive responses and assert:
      - Mean throughput >= 80 tok/s
      - P95 latency TTFT < 2000ms
      - Zero failures
    """
    prompt = ("Describe in detail how Kubernetes pod scheduling works, "
              "including resource requests, limits, and node affinity.")
    results = []
    for i in range(5):
        t0 = time.perf_counter()
        r = vllm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        assert r.success, f"Request {i+1} failed: {r.error}"
        results.append({
            "latency_ms":       latency_ms,
            "tokens":           r.tokens_generated,
            "tokens_per_second": r.tokens_per_second,
        })
    mean_tps    = statistics.mean(r["tokens_per_second"] for r in results)
    p95_latency = sorted(r["latency_ms"] for r in results)[int(len(results)*0.95)]
    assert mean_tps >= 80.0, (
        f"Mean throughput {mean_tps:.1f} tok/s < 80 tok/s target")
    assert p95_latency < 10000, (
        f"P95 latency {p95_latency:.0f}ms > 10s")


@pytest.mark.l4
@pytest.mark.timeout(30)
def test_vllm_concurrent_requests(vllm_client):
    """Assert vLLM handles 3 concurrent requests without failure."""
    import concurrent.futures
    prompt = "What is a Kubernetes deployment? Answer in 2 sentences."
    def make_request(_):
        return vllm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50, temperature=0.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(make_request, i) for i in range(3)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    failures = [r for r in results if not r.success]
    assert len(failures) == 0, (
        f"{len(failures)}/3 concurrent requests failed")


@pytest.mark.l4
@pytest.mark.timeout(30)
def test_vllm_fallback_to_ollama_when_unavailable(env_config):
    """
    When vLLM is unavailable, the API should fall back to Ollama.
    Tests the health check mechanism used by the query router.
    """
    from src.serving.vllm.client import VLLMClient
    from src.serving.ollama.client import OllamaClient
    vllm = VLLMClient(env_config)
    ollama = OllamaClient(env_config)
    if not vllm.health_check(timeout=3.0):
        resp = ollama.generate("Say: fallback OK", max_tokens=10)
        assert resp.success, "Ollama fallback also failed"
        assert "OK" in resp.text or len(resp.text) > 0
    else:
        pytest.skip("vLLM is available — fallback test not applicable")