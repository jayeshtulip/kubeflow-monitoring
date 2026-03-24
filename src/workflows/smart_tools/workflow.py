"""
Smart Tools workflow.
Conditionally selects the right tool based on query classification
before calling the LLM. Skips tools with low relevance.
Average latency: ~33s.
Used when the system detects query is tool-heavy but not complex enough for PEC.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from src.serving.ollama.client import OllamaClient
from src.storage.qdrant.retriever import multi_collection_search
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

SMART_SYSTEM_PROMPT = """You are an expert infrastructure assistant.
Answer the question using the provided context from multiple sources.
Cite the source type (knowledge base, logs, tickets) when relevant.
Be specific and actionable."""


@dataclass
class SmartToolsResult:
    query: str
    response: str
    tools_used: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    workflow: str = "smart_tools"


_TOOL_KEYWORDS = {
    "cloudwatch": ["error", "log", "exception", "traceback", "crash",
                   "timeout", "latency", "spike", "metric"],
    "jira":       ["incident", "ticket", "issue", "bug", "outage",
                   "previous", "similar", "history", "reported"],
    "confluence":  ["runbook", "procedure", "how to", "policy", "guide",
                   "documentation", "setup", "configure", "process"],
    "kubectl":    ["pod", "node", "deployment", "namespace", "event",
                   "eviction", "crash", "restart", "pending"],
}


def _select_tools(query: str) -> list[str]:
    """Select tools relevant to the query based on keyword matching."""
    q = query.lower()
    selected = ["qdrant"]  # always search knowledge base
    for tool, keywords in _TOOL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            selected.append(tool)
    return selected


def _gather_context(
    query: str,
    tools: list[str],
    cfg: EnvConfig,
) -> tuple[str, list[str]]:
    """Run selected tools and aggregate context. Returns (context_text, tools_used)."""
    context_parts: list[str] = []
    tools_used: list[str] = []

    if "qdrant" in tools:
        chunks = multi_collection_search(
            query, ["tech_docs", "hr_policies", "org_info"],
            top_k_per_collection=3, cfg=cfg)
        if chunks:
            kb_text = "\n".join(
                f"[KB | {c.source} | {c.score:.2f}] {c.text[:200]}"
                for c in chunks[:5])
            context_parts.append(f"Knowledge Base:\n{kb_text}")
            tools_used.append("qdrant")

    if "cloudwatch" in tools:
        from src.agents.tools.cloudwatch import query_cloudwatch_logs
        logs = query_cloudwatch_logs(
            "/aws/eks/llm-platform/application", query, cfg=cfg)
        if logs and not logs[0].get("error"):
            log_text = "\n".join(str(l)[:150] for l in logs[:3])
            context_parts.append(f"CloudWatch Logs:\n{log_text}")
            tools_used.append("cloudwatch")

    if "jira" in tools:
        from src.agents.tools.jira import search_jira_tickets
        tickets = search_jira_tickets(
            f'summary ~ "{query[:50]}" ORDER BY created DESC', max_results=3)
        if tickets and not tickets[0].get("error"):
            jira_text = "\n".join(
                f"[{t.get('key')}] {t.get('summary','')} ({t.get('status','')})"
                for t in tickets)
            context_parts.append(f"Jira Tickets:\n{jira_text}")
            tools_used.append("jira")

    if "confluence" in tools:
        from src.agents.tools.confluence import search_confluence
        pages = search_confluence(query, max_results=2)
        if pages and not pages[0].get("error"):
            conf_text = "\n".join(
                f"[{p.get('title','')}] {p.get('excerpt','')[:200]}"
                for p in pages)
            context_parts.append(f"Confluence:\n{conf_text}")
            tools_used.append("confluence")

    if "kubectl" in tools:
        from src.agents.tools.kubectl import get_pod_events
        events = get_pod_events()
        if isinstance(events, dict) and "items" in events:
            evt_text = "\n".join(
                str(e.get("message", ""))[:150]
                for e in events["items"][:3])
            context_parts.append(f"Kubernetes Events:\n{evt_text}")
            tools_used.append("kubectl")

    return "\n\n".join(context_parts), tools_used


def run(
    query: str,
    cfg: EnvConfig | None = None,
) -> SmartToolsResult:
    """Execute Smart Tools workflow: classify -> gather -> generate."""
    cfg = cfg or EnvConfig()
    t_start = time.perf_counter()

    # Select relevant tools
    tools = _select_tools(query)
    logger.info("SmartTools: selected tools=%s for query=%s", tools, query[:60])

    # Gather context
    context, tools_used = _gather_context(query, tools, cfg)

    # Generate response
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    client = OllamaClient(cfg)
    response = client.generate(
        prompt=prompt,
        system=SMART_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )

    latency = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info("SmartTools: %.0fms %d tools query=%s",
                latency, len(tools_used), query[:60])

    return SmartToolsResult(
        query=query,
        response=response.text if response.success else "",
        tools_used=tools_used,
        total_latency_ms=latency,
        success=response.success,
        error=response.error if not response.success else "",
    )