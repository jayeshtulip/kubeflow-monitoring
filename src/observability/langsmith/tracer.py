"""
LangSmith tracing integration.
Wraps LangSmith SDK to trace every agent step.
Also handles human feedback push from Chatbot UI.
"""
from __future__ import annotations
import os
import time
from typing import Any
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)


def get_langsmith_client():
    """Return LangSmith client if API key is configured."""
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    if not api_key:
        return None
    try:
        from langsmith import Client
        return Client(api_key=api_key)
    except ImportError:
        logger.warning("langsmith package not installed")
        return None


class AgentTracer:
    """
    Context manager that wraps an agent step in a LangSmith trace.
    Usage:
        with AgentTracer("planner", query=query) as tracer:
            result = planner.create_plan(query)
            tracer.set_output(result)
    """

    def __init__(
        self,
        step_name: str,
        run_id: str = "",
        parent_run_id: str = "",
        project: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.step_name     = step_name
        self.run_id        = run_id or f"{step_name}-{int(time.time())}"
        self.parent_run_id = parent_run_id
        self.project       = project or os.environ.get("LANGSMITH_PROJECT", "llm-platform-prod")
        self.metadata      = metadata or {}
        self.inputs: dict[str, Any] = kwargs
        self.outputs: dict[str, Any] = {}
        self.error: str | None = None
        self._client = get_langsmith_client()
        self._t0     = 0.0

    def set_output(self, output: Any) -> None:
        self.outputs = {"output": output} if not isinstance(output, dict) else output

    def set_error(self, error: str) -> None:
        self.error = error

    def __enter__(self) -> "AgentTracer":
        self._t0 = time.perf_counter()
        if self._client:
            try:
                self._client.create_run(
                    name=self.step_name,
                    run_type="chain",
                    inputs=self.inputs,
                    project_name=self.project,
                    id=self.run_id,
                    parent_run_id=self.parent_run_id or None,
                    extra={"metadata": self.metadata},
                )
            except Exception as exc:
                logger.debug("LangSmith create_run failed: %s", exc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        if exc_type is not None:
            self.error = str(exc_val)
        if self._client:
            try:
                self._client.update_run(
                    run_id=self.run_id,
                    outputs=self.outputs,
                    error=self.error,
                    extra={"latency_ms": latency_ms},
                    end_time=time.time(),
                )
            except Exception as exc:
                logger.debug("LangSmith update_run failed: %s", exc)
        return False


def push_human_feedback(
    run_id: str,
    score: float,
    comment: str = "",
) -> bool:
    """
    Push thumbs-up (1.0) / thumbs-down (0.0) from Chatbot UI to LangSmith.
    Returns True on success.
    """
    client = get_langsmith_client()
    if not client:
        return False
    try:
        client.create_feedback(
            run_id=run_id,
            key="user_score",
            score=score,
            comment=comment,
        )
        logger.info("Feedback pushed: run=%s score=%.1f", run_id, score)
        return True
    except Exception as exc:
        logger.error("LangSmith feedback push failed: %s", exc)
        return False