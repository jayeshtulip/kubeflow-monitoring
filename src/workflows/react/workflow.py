"""
ReAct (Reasoning + Acting) workflow.
Used for 25% of queries: multi-step reasoning with tool use,
no formal investigation plan, no Critic replan loop.
Average latency: ~35s.

Pattern: Think -> Act -> Observe -> Think -> Act -> ... -> Answer
Max 5 reasoning steps before final answer.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from src.serving.ollama.client import OllamaClient
from src.storage.qdrant.retriever import multi_collection_search
from src.agents.tools.cloudwatch import query_cloudwatch_logs
from src.agents.tools.jira import search_jira_tickets
from src.agents.tools.confluence import search_confluence
from src.agents.tools.kubectl import get_pod_events
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

MAX_STEPS = 5

REACT_SYSTEM_PROMPT = """You are an expert infrastructure engineer.
Use the ReAct pattern to answer questions:
  Thought: reason about what to do next
  Action: one of [search_kb, check_logs, search_jira, search_confluence, get_events, answer]
  Observation: result of the action

When you have enough information, use Action: answer
Keep each Thought concise. Use Action: answer when confident."""


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class ReActWorkflowResult:
    query: str
    response: str
    steps: list[ReActStep] = field(default_factory=list)
    total_latency_ms: float = 0.0
    llm_call_count: int = 0
    tool_call_count: int = 0
    success: bool = True
    error: str = ""
    workflow: str = "react"


_TOOLS = {
    "search_kb": lambda q, cfg: [
        c.text for c in multi_collection_search(
            q, ["tech_docs", "hr_policies", "org_info"],
            top_k_per_collection=3, cfg=cfg)[:5]
    ],
    "check_logs":        lambda q, cfg: query_cloudwatch_logs(
        "/aws/eks/llm-platform/application", q, cfg=cfg),
    "search_jira":       lambda q, cfg: search_jira_tickets(
        f'summary ~ "{q}" OR description ~ "{q}"'),
    "search_confluence": lambda q, cfg: search_confluence(q),
    "get_events":        lambda q, cfg: get_pod_events(),
}


def _parse_react_response(text: str) -> tuple[str, str, str]:
    """Parse Thought/Action/Action_Input from LLM response.
    Returns (thought, action, action_input).
    """
    thought, action, action_input = "", "answer", ""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            action = line.replace("Action:", "").strip().lower().replace(" ", "_")
        elif line.startswith("Action_Input:") or line.startswith("Action Input:"):
            action_input = line.split(":", 1)[-1].strip()
    return thought, action, action_input


def run(
    query: str,
    cfg: EnvConfig | None = None,
) -> ReActWorkflowResult:
    """Execute ReAct workflow with up to MAX_STEPS reasoning cycles."""
    cfg = cfg or EnvConfig()
    client = OllamaClient(cfg)
    t_start = time.perf_counter()
    steps: list[ReActStep] = []
    llm_calls = 0
    tool_calls = 0
    scratchpad = f"Question: {query}\n"

    for step_num in range(MAX_STEPS):
        prompt = scratchpad + "Thought:"
        response = client.generate(
            prompt=prompt,
            system=REACT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256,
            timeout=60.0,
        )
        llm_calls += 1

        if not response.success:
            break

        thought, action, action_input = _parse_react_response(
            "Thought: " + response.text)

        # Terminal action
        if action == "answer" or step_num == MAX_STEPS - 1:
            steps.append(ReActStep(
                thought=thought, action="answer",
                action_input="", observation=""))
            scratchpad += f"Thought: {thought}\nAction: answer\n"
            break

        # Execute tool
        tool_fn = _TOOLS.get(action)
        if tool_fn:
            try:
                observation = tool_fn(action_input or query, cfg)
                if isinstance(observation, list):
                    obs_text = "\n".join(
                        str(o)[:200] for o in observation[:3])
                else:
                    obs_text = str(observation)[:500]
                tool_calls += 1
            except Exception as exc:
                obs_text = f"Tool error: {exc}"
        else:
            obs_text = f"Unknown action: {action}"

        steps.append(ReActStep(
            thought=thought, action=action,
            action_input=action_input, observation=obs_text))
        scratchpad += (
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action_Input: {action_input}\n"
            f"Observation: {obs_text}\n"
        )

    # Final answer generation
    final_prompt = (
        scratchpad
        + "\nBased on the above reasoning, provide a clear final answer:"
    )
    final = client.generate(
        prompt=final_prompt,
        system=REACT_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )
    llm_calls += 1
    response_text = final.text if final.success else scratchpad

    latency = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info("ReAct: %d steps %d tool_calls %.0fms query=%s",
                len(steps), tool_calls, latency, query[:60])

    return ReActWorkflowResult(
        query=query,
        response=response_text,
        steps=steps,
        total_latency_ms=latency,
        llm_call_count=llm_calls,
        tool_call_count=tool_calls,
        success=True,
    )