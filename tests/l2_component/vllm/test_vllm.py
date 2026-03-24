"""L2 Component tests — vLLM inference server on H100."""
import pytest


@pytest.mark.l2
@pytest.mark.timeout(10)
def test_vllm_health_endpoint(vllm_client):
    """GET /health returns True within 5s."""
    ok = vllm_client.health_check(timeout=5.0)
    assert ok, "vLLM /health endpoint not responding"


@pytest.mark.l2
@pytest.mark.timeout(60)
def test_vllm_generate_response(vllm_client):
    """Generate a short response and assert content is non-empty."""
    response = vllm_client.chat(
        messages=[{"role": "user", "content": "Say: OK"}],
        max_tokens=10,
        temperature=0.0,
    )
    assert response.success, f"vLLM generation failed: {response.error}"
    assert len(response.text) > 0, "vLLM returned empty response"


@pytest.mark.l2
@pytest.mark.timeout(120)
def test_vllm_throughput_above_threshold(vllm_client):
    """Assert vLLM generates >= 80 tokens/second on H100."""
    response = vllm_client.chat(
        messages=[{"role": "user",
                   "content": "Describe Kubernetes pod scheduling in detail."}],
        max_tokens=200,
        temperature=0.0,
    )
    assert response.success, f"vLLM generation failed: {response.error}"
    assert response.tokens_per_second >= 80.0, (
        f"vLLM throughput {response.tokens_per_second:.1f} tok/s < 80 tok/s"
    )