"""
Qdrant document indexer.

Responsibilities:
  1. Chunk a raw text document into overlapping windows
  2. Generate sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dims)
  3. Upsert vectors into a named Qdrant collection with rich metadata
  4. Return indexing metrics (chunk count, upsert latency, mean vector norm)

Used by:
  - Kubeflow Pipeline 05 (Data Indexing)
  - src/api/routers/document_upload.py (real-time ad-hoc indexing)
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

from pipelines.components.shared.base import (
    EnvConfig,
    ensure_qdrant_collection,
    get_logger,
    get_qdrant_client,
)

logger = get_logger(__name__)

# Singleton model — loaded once per process/pod
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        logger.info("Loading embedding model: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# ── Chunking ──────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_word: int
    end_word: int
    doc_id: str
    source: str
    collection: str
    extra_metadata: dict[str, Any]


def chunk_text(
    text: str,
    doc_id: str,
    source: str,
    collection: str,
    chunk_size: int = 150,
    overlap: int = 30,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping word-windows.

    Args:
        text:           Raw document text.
        doc_id:         Stable document identifier (e.g. Confluence page ID).
        source:         Human-readable source label (e.g. "confluence", "jira").
        collection:     Target Qdrant collection name.
        chunk_size:     Target chunk size in words (Katib-tuned default: 150).
        overlap:        Word overlap between consecutive chunks (default: 30).
        extra_metadata: Any additional fields to store alongside each chunk.

    Returns:
        List of Chunk objects ready for embedding.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty document text")

    words = text.split()
    if not words:
        raise ValueError("Document produced zero words after split")

    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)

    for i, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        # Skip very short trailing chunks (< 10 words)
        if len(chunk_words) < 10 and i > 0:
            break

        chunks.append(
            Chunk(
                text=" ".join(chunk_words),
                chunk_index=i,
                start_word=start,
                end_word=end,
                doc_id=doc_id,
                source=source,
                collection=collection,
                extra_metadata=extra_metadata or {},
            )
        )

    logger.info(
        "Chunked doc '%s': %d words → %d chunks (size=%d, overlap=%d)",
        doc_id,
        len(words),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> list[tuple[Chunk, list[float]]]:
    """
    Generate embeddings for a list of chunks in batches.

    Returns:
        List of (chunk, embedding_vector) tuples.
    """
    model = _get_model(model_name)
    texts = [c.text for c in chunks]
    t0 = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit-norm → cosine = dot product
        convert_to_numpy=True,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Embedded %d chunks in %.2fs (%.1f chunks/s)",
        len(chunks),
        elapsed,
        len(chunks) / elapsed if elapsed > 0 else 0,
    )
    return list(zip(chunks, [e.tolist() for e in embeddings]))


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_to_qdrant(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    qdrant_client: QdrantClient,
    batch_size: int = 128,
) -> dict[str, Any]:
    """
    Upsert (chunk, vector) pairs into Qdrant.

    Point ID is a deterministic UUID derived from doc_id + chunk_index,
    so re-indexing the same document overwrites existing points cleanly.

    Returns:
        Dict of upsert metrics.
    """
    if not chunk_embeddings:
        raise ValueError("No chunk embeddings provided for upsert")

    collection = chunk_embeddings[0][0].collection
    ensure_qdrant_collection(qdrant_client, collection, dims=len(chunk_embeddings[0][1]))

    points: list[PointStruct] = []
    for chunk, vector in chunk_embeddings:
        # Deterministic ID: same doc + chunk always produces same UUID
        point_id = str(
            uuid.UUID(
                hashlib.md5(
                    f"{chunk.doc_id}::{chunk.chunk_index}".encode()
                ).hexdigest()
            )
        )
        payload = {
            "text":        chunk.text,
            "doc_id":      chunk.doc_id,
            "source":      chunk.source,
            "collection":  chunk.collection,
            "chunk_index": chunk.chunk_index,
            "start_word":  chunk.start_word,
            "end_word":    chunk.end_word,
            **chunk.extra_metadata,
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    t0 = time.perf_counter()
    total_upserted = 0

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant_client.upsert(collection_name=collection, points=batch)
        total_upserted += len(batch)
        logger.debug("Upserted batch %d/%d", i // batch_size + 1, -(-len(points) // batch_size))

    elapsed = time.perf_counter() - t0
    metrics = {
        "collection":      collection,
        "chunks_upserted": total_upserted,
        "upsert_seconds":  round(elapsed, 3),
        "upsert_rate":     round(total_upserted / elapsed, 1) if elapsed > 0 else 0,
    }
    logger.info("Upsert complete: %s", metrics)
    return metrics


# ── High-level entry point ────────────────────────────────────────────────────

@dataclass
class IndexingResult:
    doc_id: str
    collection: str
    chunk_count: int
    upsert_metrics: dict[str, Any]
    total_seconds: float
    success: bool
    error: str = ""


def index_document(
    text: str,
    doc_id: str,
    source: str,
    collection: str,
    cfg: EnvConfig | None = None,
    qdrant_client: QdrantClient | None = None,
    chunk_size: int = 150,
    overlap: int = 30,
    extra_metadata: dict[str, Any] | None = None,
) -> IndexingResult:
    """
    Full pipeline: text → chunks → embeddings → Qdrant upsert.

    Can accept an existing qdrant_client (for testing / reuse) or will
    create one from cfg.
    """
    cfg = cfg or EnvConfig()
    client = qdrant_client or get_qdrant_client(cfg)
    t_start = time.perf_counter()

    try:
        chunks = chunk_text(
            text=text,
            doc_id=doc_id,
            source=source,
            collection=collection,
            chunk_size=chunk_size,
            overlap=overlap,
            extra_metadata=extra_metadata,
        )
        chunk_embeddings = embed_chunks(chunks, model_name=cfg.embedding_model)
        upsert_metrics = upsert_to_qdrant(chunk_embeddings, client)

        return IndexingResult(
            doc_id=doc_id,
            collection=collection,
            chunk_count=len(chunks),
            upsert_metrics=upsert_metrics,
            total_seconds=round(time.perf_counter() - t_start, 3),
            success=True,
        )

    except Exception as exc:
        logger.exception("Indexing failed for doc '%s': %s", doc_id, exc)
        return IndexingResult(
            doc_id=doc_id,
            collection=collection,
            chunk_count=0,
            upsert_metrics={},
            total_seconds=round(time.perf_counter() - t_start, 3),
            success=False,
            error=str(exc),
        )
