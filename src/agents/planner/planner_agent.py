"""
Planner agent — generates a systematic investigation plan.

Given a query, produces a numbered list of investigation steps
with: what to check, which tool to use, what evidence to gather.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from src.serving.ollama.client import OllamaClient
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a senior SRE (Site Reliability Engineer) and expert investigator.
Your job is to create a systematic investigation plan for infrastructure incidents.

Given a query about an infrastructure issue, create a numbered investigation plan.
Each step must specify:
  - What to check
  - Which tool to use (cloudwatch_logs, cloudwatch_metrics, kubectl, jira, confluence, qdrant_search)
  - What evidence you expect to find

Format each step as:
Step N: [Action] using [tool] - [expected evidence]

Keep plans focused: 3-7 steps maximum.
Think about causal chains: what caused what?
Always include a final correlation/root-cause step."""


@dataclass
class InvestigationPlan:
    query: str
    steps: list[dict] = field(default_factory=list)
    raw_plan: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""

    def to_prompt_context(self) -> str:
        """Format plan as context for the Executor agent."""
        lines = [f"Investigation plan for: {self.query}", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.get('action', '')} using {step.get('tool', '')} - {step.get('expected', '')}")
        return "\n".join(lines)


def _parse_plan(raw: str, query: str) -> InvestigationPlan:
    """Parse free-text plan into structured steps."""
    steps = []
    lines = raw.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or not (line[0].isdigit() or line.lower().startswith("step")):
            continue
        # Extract tool mention
        tool = "qdrant_search"
        for t in ["cloudwatch_logs", "cloudwatch_metrics", "kubectl",
                  "jira", "confluence", "qdrant_search"]:
            if t.lower() in line.lower():
                tool = t
                break
        steps.append({
            "action":   line,
            "tool":     tool,
            "expected": "",
            "done":     False,
            "result":   None,
        })
    # Fallback if parsing failed
    if not steps:
        steps = [
            {"action": "Search knowledge base", "tool": "qdrant_search",
             "expected": "Relevant documentation", "done": False, "result": None},
            {"action": "Check recent incidents", "tool": "jira",
             "expected": "Related tickets", "done": False, "result": None},
        ]
    return InvestigationPlan(query=query, steps=steps, raw_plan=raw)


def create_plan(
    query: str,
    critic_feedback: str = "",
    cfg: EnvConfig | None = None,
) -> InvestigationPlan:
    """
    Generate an investigation plan for the given query.
    If critic_feedback is provided, this is a replan.
    """
    cfg = cfg or EnvConfig()
    client = OllamaClient(cfg)

    if critic_feedback:
        prompt = (
            f"Original query: {query}\n\n"
            f"Previous investigation was incomplete. Critic feedback:\n{critic_feedback}\n\n"
            f"Create an updated investigation plan addressing the gaps above."
        )
    else:
        prompt = (
            f"Query: {query}\n\n"
            f"Create a systematic investigation plan to diagnose and resolve this issue."
        )

    response = client.generate(
        prompt=prompt,
        system=PLANNER_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=512,
    )

    if not response.success:
        logger.error("Planner failed: %s", response.error)
        plan = InvestigationPlan(
            query=query, success=False, error=response.error)
        plan.steps = [
            {"action": "Search knowledge base", "tool": "qdrant_search",
             "expected": "Relevant context", "done": False, "result": None}
        ]
        return plan

    plan = _parse_plan(response.text, query)
    plan.latency_ms = response.latency_ms
    logger.info("Planner: %d steps in %.0fms for query=%s",
                len(plan.steps), plan.latency_ms, query[:60])
    return plan