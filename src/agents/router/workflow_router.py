"""
Workflow router — classifies query complexity and routes to correct workflow.

Routing rules:
  Simple  (70%): single-domain lookup, factual, < 5 words typical
  ReAct   (25%): multi-step reasoning, requires tools but no re-planning
  PEC      (5%): complex incident investigation, causal chain needed
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)


class WorkflowType(str, Enum):
    SIMPLE = "simple"
    TOOLS  = "tools"
    SMART  = "smart"
    REACT  = "react"
    PLANNER_EXECUTOR_CRITIC = "planner_executor_critic"


@dataclass
class RoutingDecision:
    workflow: WorkflowType
    confidence: float
    reason: str
    complexity_score: int  # 1-10


# Signal words for complexity scoring
HIGH_COMPLEXITY = [
    "why", "root cause", "intermittent", "timeout", "crash", "failing",
    "outage", "investigate", "debug", "not working", "keeps", "randomly",
    "spike", "degraded", "slow", "latency", "memory", "cpu", "eviction",
    "connection refused", "error", "exception", "traceback",
]

MEDIUM_COMPLEXITY = [
    "how do i", "how to", "configure", "setup", "deploy", "restart",
    "scale", "update", "check", "monitor", "logs for", "status of",
]

LOW_COMPLEXITY = [
    "what is", "who is", "when is", "list", "show me", "what are",
    "policy", "procedure", "contact", "oncall", "schedule",
]


def route_query(query: str) -> RoutingDecision:
    """
    Classify query complexity and return routing decision.
    Uses keyword scoring with length heuristics.
    """
    q = query.lower().strip()
    words = q.split()
    score = 5  # baseline

    high_hits   = sum(1 for s in HIGH_COMPLEXITY if s in q)
    medium_hits = sum(1 for s in MEDIUM_COMPLEXITY if s in q)
    low_hits    = sum(1 for s in LOW_COMPLEXITY if s in q)

    score += high_hits * 2
    score += medium_hits
    score -= low_hits

    # Length heuristic: longer queries tend to be more complex
    if len(words) > 15:
        score += 2
    elif len(words) < 6:
        score -= 2

    # Multiple symptoms = more complex
    if q.count("and") + q.count("also") + q.count("plus") > 1:
        score += 1

    score = max(1, min(10, score))

    if score >= 7:
        workflow = WorkflowType.PLANNER_EXECUTOR_CRITIC
        confidence = min(0.95, 0.7 + (score - 7) * 0.08)
        reason = f"High complexity (score={score}): investigation required"
    elif score >= 4:
        workflow = WorkflowType.REACT
        confidence = min(0.90, 0.65 + (score - 4) * 0.05)
        reason = f"Medium complexity (score={score}): multi-step reasoning"
    else:
        workflow = WorkflowType.SIMPLE
        confidence = min(0.95, 0.8 + (3 - score) * 0.05)
        reason = f"Low complexity (score={score}): direct lookup"

    decision = RoutingDecision(
        workflow=workflow,
        confidence=confidence,
        reason=reason,
        complexity_score=score,
    )
    logger.info("Router: %s (score=%d conf=%.2f) for query=%s",
                workflow.value, score, confidence, query[:60])
    return decision