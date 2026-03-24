"""
Custom Prometheus metrics for the LLM Platform.
All metrics are registered at module import time.
Exposed at /metrics via prometheus-client WSGI app.
"""
from __future__ import annotations
from prometheus_client import (
    Counter, Histogram, Gauge,
    CollectorRegistry, generate_latest,
)
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)

REGISTRY = CollectorRegistry()

# ── Request metrics ───────────────────────────────────────────────────────

REQUEST_TOTAL = Counter(
    "llm_requests_total",
    "Total query requests",
    ["workflow_type", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION_MS = Histogram(
    "llm_request_duration_ms",
    "Request latency in milliseconds",
    ["workflow_type"],
    buckets=[100, 500, 1000, 5000, 10000, 20000, 30000, 60000, 90000],
    registry=REGISTRY,
)

TOOL_CALLS_TOTAL = Counter(
    "llm_tool_calls_total",
    "Total tool calls by executor agent",
    ["tool_name"],
    registry=REGISTRY,
)

REPLAN_TOTAL = Counter(
    "llm_replan_total",
    "Total Planner-Executor-Critic replan cycles",
    registry=REGISTRY,
)

# ── Guardrail metrics ─────────────────────────────────────────────────────

GUARDRAIL_TRIGGERED = Counter(
    "guardrail_triggered_total",
    "Guardrail trigger count by type",
    ["trigger_type"],
    registry=REGISTRY,
)

# ── Quality metrics ───────────────────────────────────────────────────────

HALLUCINATION_RATE = Gauge(
    "llm_hallucination_rate",
    "Current hallucination rate from latest P09 RAGAS evaluation",
    registry=REGISTRY,
)

RAGAS_FAITHFULNESS = Gauge(
    "llm_ragas_faithfulness",
    "Current RAGAS faithfulness score",
    registry=REGISTRY,
)

PLATFORM_HEALTH_SCORE = Gauge(
    "llm_platform_health_score",
    "Platform health score 0-1 from P04 quality monitoring",
    registry=REGISTRY,
)

# ── Drift metrics ─────────────────────────────────────────────────────────

DATA_DRIFT_SCORE = Gauge(
    "llm_data_drift_js_divergence",
    "Jensen-Shannon divergence of query embedding distributions",
    registry=REGISTRY,
)

# ── Helper functions ──────────────────────────────────────────────────────

def record_request(
    workflow_type: str,
    latency_ms: float,
    success: bool,
    tool_calls: int = 0,
    replan_count: int = 0,
) -> None:
    """Record metrics for a completed API request."""
    status = "success" if success else "error"
    REQUEST_TOTAL.labels(workflow_type=workflow_type, status=status).inc()
    REQUEST_DURATION_MS.labels(workflow_type=workflow_type).observe(latency_ms)
    if replan_count > 0:
        REPLAN_TOTAL.inc(replan_count)


def record_tool_call(tool_name: str) -> None:
    TOOL_CALLS_TOTAL.labels(tool_name=tool_name).inc()


def record_guardrail_trigger(trigger_type: str) -> None:
    GUARDRAIL_TRIGGERED.labels(trigger_type=trigger_type).inc()


def update_quality_metrics(
    hallucination_rate: float,
    faithfulness: float,
    health_score: float,
) -> None:
    HALLUCINATION_RATE.set(hallucination_rate)
    RAGAS_FAITHFULNESS.set(faithfulness)
    PLATFORM_HEALTH_SCORE.set(health_score)


def get_metrics_output() -> bytes:
    """Return Prometheus text exposition format."""
    return generate_latest(REGISTRY)