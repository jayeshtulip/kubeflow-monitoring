"""
Executor agent — executes each step of the investigation plan.
Calls tools dynamically based on step.tool field.
Maintains execution context across steps.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from src.agents.planner.planner_agent import InvestigationPlan
from src.agents.tools.qdrant_search import search_knowledge_base
from src.agents.tools.cloudwatch import query_cloudwatch_logs, query_cloudwatch_metrics
from src.agents.tools.jira import search_jira_tickets
from src.agents.tools.confluence import search_confluence
from src.agents.tools.kubectl import run_kubectl, get_pod_events
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)


TOOL_REGISTRY = {
    "qdrant_search":      lambda q, cfg: search_knowledge_base(q, cfg=cfg),
    "cloudwatch_logs":    lambda q, cfg: query_cloudwatch_logs(
                              "/aws/eks/llm-platform/application", q, cfg=cfg),
    "cloudwatch_metrics": lambda q, cfg: query_cloudwatch_metrics(
                              "AWS/RDS", q, cfg=cfg),
    "jira":               lambda q, cfg: search_jira_tickets(
                              f'summary ~ "{q}" OR description ~ "{q}"'),
    "confluence":         lambda q, cfg: search_confluence(q),
    "kubectl":            lambda q, cfg: get_pod_events(),
}


@dataclass
class ExecutionContext:
    query: str
    plan: InvestigationPlan
    step_results: list[dict] = field(default_factory=list)
    total_latency_ms: float = 0.0
    tool_call_count: int = 0
    success: bool = True
    error: str = ""

    def get_evidence_summary(self) -> str:
        """Summarise all gathered evidence for the Critic."""
        lines = [f"Evidence gathered for: {self.query}", ""]
        for i, result in enumerate(self.step_results, 1):
            step_action = self.plan.steps[i-1].get("action", f"Step {i}") if i <= len(self.plan.steps) else f"Step {i}"
            lines.append(f"Step {i}: {step_action}")
            if isinstance(result, list):
                for item in result[:3]:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("summary") or item.get("raw") or str(item)[:200]
                        lines.append(f"  - {text[:200]}")
            elif isinstance(result, dict):
                lines.append(f"  - {str(result)[:200]}")
            lines.append("")
        return "\n".join(lines)


def _extract_search_query(step_action: str, original_query: str) -> str:
    """Extract a usable search query from a step action description."""
    # Remove step prefix if present
    import re
    action = re.sub(r"^Step \d+:\s*", "", step_action).strip()
    # Use original query if action is too short
    if len(action) < 10:
        return original_query
    return action


def execute_plan(
    plan: InvestigationPlan,
    cfg: EnvConfig | None = None,
) -> ExecutionContext:
    """
    Execute each step in the investigation plan.
    Returns ExecutionContext with all step results.
    """
    cfg = cfg or EnvConfig()
    ctx = ExecutionContext(query=plan.query, plan=plan)
    t_start = time.perf_counter()

    for i, step in enumerate(plan.steps):
        tool_name = step.get("tool", "qdrant_search")
        action    = step.get("action", plan.query)
        search_q  = _extract_search_query(action, plan.query)

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            logger.warning("Unknown tool %s — falling back to qdrant_search", tool_name)
            tool_fn = TOOL_REGISTRY["qdrant_search"]

        t0 = time.perf_counter()
        try:
            result = tool_fn(search_q, cfg)
            step["done"]   = True
            step["result"] = result
            ctx.step_results.append(result)
            ctx.tool_call_count += 1
            latency = round((time.perf_counter() - t0) * 1000, 2)
            logger.info("Executor step %d/%d [%s]: %.0fms",
                        i + 1, len(plan.steps), tool_name, latency)
        except Exception as exc:
            logger.error("Executor step %d failed: %s", i + 1, exc)
            step["done"]   = False
            step["result"] = {"error": str(exc)}
            ctx.step_results.append({"error": str(exc)})

    ctx.total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info("Executor complete: %d steps, %d tool calls, %.0fms",
                len(plan.steps), ctx.tool_call_count, ctx.total_latency_ms)
    return ctx