"""Qdrant vector search tool for the Executor agent."""
from __future__ import annotations
from src.storage.qdrant.retriever import multi_collection_search, RetrievedChunk
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

COLLECTIONS = ["tech_docs", "hr_policies", "org_info"]


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    cfg: EnvConfig | None = None,
) -> list[dict]:
    """
    Search all Qdrant collections for relevant context.
    Returns list of dicts with text, score, source, doc_id.
    """
    cfg = cfg or EnvConfig()
    chunks = multi_collection_search(
        query=query,
        collections=COLLECTIONS,
        top_k_per_collection=top_k,
        cfg=cfg,
    )
    results = [
        {
            "text":    c.text,
            "score":   c.score,
            "source":  c.source,
            "doc_id":  c.doc_id,
            "tool":    "qdrant_search",
        }
        for c in chunks[:top_k]
    ]
    logger.info("qdrant_search: %d results for query=%s", len(results), query[:60])
    return results