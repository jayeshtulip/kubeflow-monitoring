"""
Planner-Executor-Critic (PEC) workflow.
Used for 5% of complex queries: incident investigation, root cause analysis.
Average latency: ~55s. Up to 2 replan cycles.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from src.agents.router.workflow_router import RoutingDecision
from src.agents.planner.planner_agent import create_plan, InvestigationPlan
from src.agents.executor.executor_agent import execute_plan, ExecutionContext
from src.agents.critic.critic_agent import validate_investigation, CriticVerdict
from src.serving.ollama.client import OllamaClient
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

MAX_REPLANS = 2

RESPONSE_SYSTEM_PROMPT = """You are a senior SRE providing a Root Cause Analysis (RCA).
Based on the investigation evidence, provide:
1. Timeline of events
2. Root cause (with evidence)
3. Immediate remediation steps
4. Long-term prevention measures
Be specific, reference actual evidence from the investigation."""


@dataclass
class PECWorkflowResult:
    query: str
    response: str
    plan: InvestigationPlan | None
    execution_ctx: ExecutionContext | None
    critic_verdict: CriticVerdict | None
    replan_count: int
    total_latency_ms: float
    llm_call_count: int
    tool_call_count: int
    success: bool
    error: str = ""
    workflow: str = "planner_executor_critic"


def _generate_response(
    query: str,
    ctx: ExecutionContext,
    verdict: CriticVerdict,
    cfg: EnvConfig,
) -> str:
    """Generate final RCA response from execution context."""
    client = OllamaClient(cfg)
    evidence = ctx.get_evidence_summary()
    prompt = (
        f"Query: {query}\n\n"
        f"Root cause identified: {verdict.root_cause}\n\n"
        f"Investigation evidence:\n{evidence[:3000]}\n\n"
        f"Provide a complete RCA with timeline, root cause, remediation, and prevention."
    )
    response = client.generate(
        prompt=prompt,
        system=RESPONSE_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.text if response.success else evidence


def run(
    query: str,
    cfg: EnvConfig | None = None,
) -> PECWorkflowResult:
    """Execute PEC workflow with up to MAX_REPLANS replan cycles."""
    cfg = cfg or EnvConfig()
    t_start = time.perf_counter()
    llm_calls = 0
    replan_count = 0
    critic_feedback = ""

    # Phase 1: Plan
    plan = create_plan(query, cfg=cfg)
    llm_calls += 1

    ctx: ExecutionContext | None = None
    verdict: CriticVerdict | None = None

    while replan_count <= MAX_REPLANS:
        # Phase 2: Execute
        ctx = execute_plan(plan, cfg=cfg)

        # Phase 3: Critic
        verdict = validate_investigation(
            ctx,
            max_replans=MAX_REPLANS,
            current_replan_count=replan_count,
            cfg=cfg,
        )
        llm_calls += 1

        if not verdict.replan_needed:
            logger.info("PEC: critic approved after %d replans", replan_count)
            break

        # Replan
        replan_count += 1
        logger.info("PEC: replan %d/%d — %s",
                    replan_count, MAX_REPLANS, verdict.feedback_for_planner)
        plan = create_plan(query, critic_feedback=verdict.feedback_for_planner, cfg=cfg)
        llm_calls += 1

    # Phase 4: Response generation
    final_response = _generate_response(query, ctx, verdict, cfg)
    llm_calls += 1

    total_latency = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "PEC complete: %.0fms | %d LLM calls | %d tool calls | %d replans",
        total_latency, llm_calls,
        ctx.tool_call_count if ctx else 0,
        replan_count,
    )

    return PECWorkflowResult(
        query=query,
        response=final_response,
        plan=plan,
        execution_ctx=ctx,
        critic_verdict=verdict,
        replan_count=replan_count,
        total_latency_ms=total_latency,
        llm_call_count=llm_calls,
        tool_call_count=ctx.tool_call_count if ctx else 0,
        success=True,
    )