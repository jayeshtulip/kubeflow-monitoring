"""
Simple Research workflow — single RAG lookup.
Used for 70% of queries: factual, policy, direct lookup.
Average latency: ~21s.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from src.storage.qdrant.retriever import multi_collection_search
from src.serving.ollama.client import OllamaClient
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

SIMPLE_SYSTEM_PROMPT = """You are a helpful infrastructure assistant.
Answer the question based on the provided context.
Be concise, accurate, and cite the source when possible.
If the context does not contain enough information, say so clearly."""


@dataclass
class SimpleWorkflowResult:
    query: str
    response: str
    contexts_used: int
    latency_ms: float
    success: bool
    error: str = ""
    workflow: str = "simple_research"


def run(
    query: str,
    top_k: int = 5,
    cfg: EnvConfig | None = None,
) -> SimpleWorkflowResult:
    """Execute simple research workflow: retrieve + generate."""
    cfg = cfg or EnvConfig()
    t_start = time.perf_counter()

    # Retrieve context
    chunks = multi_collection_search(
        query=query,
        collections=["tech_docs", "hr_policies", "org_info"],
        top_k_per_collection=top_k,
        cfg=cfg,
    )

    if not chunks:
        return SimpleWorkflowResult(
            query=query,
            response="I could not find relevant information for your query.",
            contexts_used=0,
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            success=True,
        )

    context = "\n\n".join(
        f"[Source: {c.source} | Score: {c.score:.2f}]\n{c.text}"
        for c in chunks[:top_k]
    )

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    client = OllamaClient(cfg)
    response = client.generate(
        prompt=prompt,
        system=SIMPLE_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )

    latency = round((time.perf_counter() - t_start) * 1000, 2)
    if response.success:
        logger.info("SimpleResearch: %.0fms, %d contexts, query=%s",
                    latency, len(chunks), query[:60])
        return SimpleWorkflowResult(
            query=query, response=response.text,
            contexts_used=len(chunks), latency_ms=latency, success=True)
    else:
        return SimpleWorkflowResult(
            query=query, response="",
            contexts_used=len(chunks), latency_ms=latency,
            success=False, error=response.error)