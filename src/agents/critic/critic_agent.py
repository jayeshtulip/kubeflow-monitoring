"""
Critic agent — validates investigation quality and decides if replan is needed.

Checks:
  - All plan steps completed
  - Evidence supports conclusions
  - Causal chain is complete
  - Root cause clearly identified
  - No gaps in the investigation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from src.serving.ollama.client import OllamaClient
from src.agents.executor.executor_agent import ExecutionContext
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a rigorous quality reviewer for infrastructure incident investigations.
Your job is to validate whether an investigation is complete and accurate.

Review the investigation results and answer:
1. Are all symptoms explained by the evidence?
2. Is there a complete causal chain (A caused B caused C)?
3. Is the root cause clearly identified with evidence?
4. Are there unexplained gaps?

Respond with EXACTLY this format:
VERDICT: COMPLETE or INCOMPLETE
CAUSAL_CHAIN: [yes/no] - brief explanation
GAPS: [list any missing investigation steps, or "none"]
ROOT_CAUSE: [one sentence root cause, or "unclear"]
RECOMMENDATION: [what additional steps are needed, if any]"""


@dataclass
class CriticVerdict:
    complete: bool
    causal_chain_valid: bool
    gaps: list[str] = field(default_factory=list)
    root_cause: str = ""
    recommendation: str = ""
    raw_verdict: str = ""
    latency_ms: float = 0.0
    replan_needed: bool = False
    feedback_for_planner: str = ""


def _parse_verdict(raw: str) -> CriticVerdict:
    """Parse critic LLM response into structured verdict."""
    import re
    verdict = CriticVerdict(complete=False, causal_chain_valid=False, raw_verdict=raw)

    lines = raw.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict.complete = "COMPLETE" in line.upper()
        elif line.startswith("CAUSAL_CHAIN:"):
            verdict.causal_chain_valid = "yes" in line.lower()
        elif line.startswith("GAPS:"):
            gaps_text = line.replace("GAPS:", "").strip()
            if gaps_text.lower() not in ("none", "none.", "-"):
                verdict.gaps = [g.strip() for g in gaps_text.split(",") if g.strip()]
        elif line.startswith("ROOT_CAUSE:"):
            verdict.root_cause = line.replace("ROOT_CAUSE:", "").strip()
        elif line.startswith("RECOMMENDATION:"):
            verdict.recommendation = line.replace("RECOMMENDATION:", "").strip()

    verdict.replan_needed = not verdict.complete or bool(verdict.gaps)
    if verdict.replan_needed:
        verdict.feedback_for_planner = (
            f"Investigation incomplete. Gaps: {verdict.gaps}. "
            f"Recommendation: {verdict.recommendation}"
        )
    return verdict


def validate_investigation(
    ctx: ExecutionContext,
    max_replans: int = 2,
    current_replan_count: int = 0,
    cfg: EnvConfig | None = None,
) -> CriticVerdict:
    """
    Validate investigation completeness.
    Returns CriticVerdict with decision and feedback.
    """
    cfg = cfg or EnvConfig()
    client = OllamaClient(cfg)

    # Hard stop: too many replans
    if current_replan_count >= max_replans:
        logger.warning("Max replans (%d) reached — forcing COMPLETE", max_replans)
        return CriticVerdict(
            complete=True,
            causal_chain_valid=True,
            root_cause="Max replan limit reached — best available answer",
            replan_needed=False,
        )

    evidence_summary = ctx.get_evidence_summary()
    prompt = (
        f"Original query: {ctx.query}\n\n"
        f"Investigation plan:\n{ctx.plan.to_prompt_context()}\n\n"
        f"Evidence gathered:\n{evidence_summary[:3000]}\n\n"
        f"Validate this investigation."
    )

    import time
    t0 = time.perf_counter()
    response = client.generate(
        prompt=prompt,
        system=CRITIC_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )
    latency = round((time.perf_counter() - t0) * 1000, 2)

    if not response.success:
        logger.error("Critic LLM failed: %s", response.error)
        return CriticVerdict(
            complete=True,
            causal_chain_valid=False,
            root_cause="Critic unavailable — proceeding with best answer",
            replan_needed=False,
            latency_ms=latency,
        )

    verdict = _parse_verdict(response.text)
    verdict.latency_ms = latency
    logger.info("Critic: complete=%s replan=%s gaps=%s latency=%.0fms",
                verdict.complete, verdict.replan_needed, verdict.gaps, latency)
    return verdict