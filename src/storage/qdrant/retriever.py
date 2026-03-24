"""
Qdrant retriever.

Responsibilities:
  1. Embed a query string using the same model used at indexing time
  2. Search a Qdrant collection for the top-k nearest neighbours
  3. Apply optional metadata filters (source, doc_id, collection)
  4. Return ranked RetrievedChunk objects consumed by RAGAS and LangGraph agents

Used by:
  - LangGraph Executor agent (tool: qdrant_search)
  - Kubeflow Pipeline 09 (RAGAS Evaluation) — context retrieval step
  - Kubeflow Pipeline 02 (RAG Optimization) — chunk strategy comparison
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from pipelines.components.shared.base import (
    EnvConfig,
    get_logger,
    get_qdrant_client,
)
from src.storage.qdrant.indexer import _get_model

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    score: float          # cosine similarity [0, 1]
    doc_id: str
    source: str
    chunk_index: int
    point_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    collection: str
    chunks: list[RetrievedChunk]
    top_k: int
    latency_ms: float
    success: bool
    error: str = ""

    @property
    def contexts(self) -> list[str]:
        """Flat list of chunk texts — format expected by RAGAS."""
        return [c.text for c in self.chunks]

    @property
    def mean_score(self) -> float:
        if not self.chunks:
            return 0.0
        return round(sum(c.score for c in self.chunks) / len(self.chunks), 4)


# ── Metadata filter builder ───────────────────────────────────────────────────

def _build_filter(
    source: str | None = None,
    doc_id: str | None = None,
) -> Filter | None:
    conditions = []
    if source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if doc_id:
        conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if not conditions:
        return None
    return Filter(must=conditions)


# ── Core search ───────────────────────────────────────────────────────────────

def search(
    query: str,
    collection: str,
    top_k: int = 5,
    source_filter: str | None = None,
    doc_id_filter: str | None = None,
    score_threshold: float = 0.0,
    cfg: EnvConfig | None = None,
    qdrant_client: QdrantClient | None = None,
) -> RetrievalResult:
    """
    Embed query and return top-k chunks from a Qdrant collection.

    Args:
        query:            Natural language query string.
        collection:       Qdrant collection name (e.g. "tech_docs").
        top_k:            Number of results to return (Katib-tuned default: 5).
        source_filter:    Optional — restrict to chunks from this source.
        doc_id_filter:    Optional — restrict to chunks from this document.
        score_threshold:  Discard results below this cosine similarity.
        cfg:              EnvConfig — uses defaults if None.
        qdrant_client:    Existing client — creates from cfg if None.

    Returns:
        RetrievalResult with ranked chunks and latency metrics.
    """
    cfg = cfg or EnvConfig()
    client = qdrant_client or get_qdrant_client(cfg)
    t0 = time.perf_counter()

    try:
        # Embed query (same model as indexing)
        model = _get_model(cfg.embedding_model)
        query_vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Build optional payload filter
        qdrant_filter = _build_filter(source=source_filter, doc_id=doc_id_filter)

        # Search
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        chunks = [
            RetrievedChunk(
                text=hit.payload.get("text", ""),
                score=round(hit.score, 4),
                doc_id=hit.payload.get("doc_id", ""),
                source=hit.payload.get("source", ""),
                chunk_index=hit.payload.get("chunk_index", -1),
                point_id=str(hit.id),
                metadata={
                    k: v
                    for k, v in hit.payload.items()
                    if k not in {"text", "doc_id", "source", "chunk_index"}
                },
            )
            for hit in hits
        ]

        logger.info(
            "search('%s', collection=%s, top_k=%d) → %d results in %.1fms, mean_score=%.3f",
            query[:60],
            collection,
            top_k,
            len(chunks),
            latency_ms,
            sum(c.score for c in chunks) / len(chunks) if chunks else 0,
        )

        return RetrievalResult(
            query_vector=query,
            collection=collection,
            chunks=chunks,
            top_k=top_k,
            latency_ms=latency_ms,
            success=True,
        )

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.exception("Retrieval failed: %s", exc)
        return RetrievalResult(
            query_vector=query,
            collection=collection,
            chunks=[],
            top_k=top_k,
            latency_ms=latency_ms,
            success=False,
            error=str(exc),
        )


# ── Multi-collection search (used by PEC executor) ───────────────────────────

def multi_collection_search(
    query: str,
    collections: list[str],
    top_k_per_collection: int = 3,
    cfg: EnvConfig | None = None,
    qdrant_client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    """
    Search multiple collections and return a merged, re-ranked result list.
    Results are sorted by score descending across all collections.
    """
    cfg = cfg or EnvConfig()
    client = qdrant_client or get_qdrant_client(cfg)
    all_chunks: list[RetrievedChunk] = []

    for col in collections:
        result = search(
            query_vector=query,
            collection=col,
            top_k=top_k_per_collection,
            cfg=cfg,
            qdrant_client=client,
        )
        if result.success:
            all_chunks.extend(result.chunks)

    # Re-rank globally by score
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    logger.info(
        "Multi-collection search across %d collections → %d total chunks",
        len(collections),
        len(all_chunks),
    )
    return all_chunks
