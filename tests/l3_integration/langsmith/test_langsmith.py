"""
L3 Integration tests — LangSmith tracing.
Verifies that agent traces are recorded in LangSmith.
Requires LANGSMITH_API_KEY env var.
"""
from __future__ import annotations
import os
import pytest

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "llm-platform-prod")

pytestmark = pytest.mark.skipif(
    not LANGSMITH_API_KEY,
    reason="LANGSMITH_API_KEY not set"
)


@pytest.fixture(scope="module")
def ls_client():
    from langsmith import Client
    return Client(api_key=LANGSMITH_API_KEY)


@pytest.mark.l3
@pytest.mark.timeout(15)
def test_langsmith_project_exists(ls_client):
    """Assert LangSmith project is accessible."""
    projects = list(ls_client.list_projects())
    names = [p.name for p in projects]
    assert LANGSMITH_PROJECT in names, (
        f"Project {LANGSMITH_PROJECT!r} not found. Available: {names}")


@pytest.mark.l3
@pytest.mark.timeout(20)
def test_agent_tracer_creates_run(ls_client):
    """AgentTracer context manager creates a run in LangSmith."""
    from src.observability.langsmith.tracer import AgentTracer
    import time
    run_id = f"test-{int(time.time())}"
    with AgentTracer("test_step", run_id=run_id, query="test query") as tracer:
        tracer.set_output({"response": "test output"})
    time.sleep(2)  # allow LangSmith to process
    run = ls_client.read_run(run_id)
    assert run is not None, f"Run {run_id} not found in LangSmith"
    assert run.name == "test_step"


@pytest.mark.l3
@pytest.mark.timeout(20)
def test_human_feedback_push(ls_client):
    """Human feedback can be pushed to LangSmith."""
    from src.observability.langsmith.tracer import push_human_feedback
    import time
    run_id = f"test-feedback-{int(time.time())}"
    with AgentTracer("test_feedback_step", run_id=run_id, query="test") as tracer:
        tracer.set_output({"response": "answer"})
    time.sleep(2)
    ok = push_human_feedback(run_id, score=1.0, comment="Good answer")
    assert ok, "push_human_feedback returned False"