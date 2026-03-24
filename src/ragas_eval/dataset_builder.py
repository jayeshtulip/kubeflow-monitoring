"""
Golden QA dataset builder.

Manages the 200-per-domain golden question-answer pairs stored in PostgreSQL.
These are consumed by:
  - Kubeflow Pipeline 09 (RAGAS Evaluation)
  - Kubeflow Pipeline 11 (Automated Retraining — post-train gate)
  - Grafana Dashboard 02 (QA coverage heatmap panel)

Schema (table: golden_qa):
  id            SERIAL PRIMARY KEY
  question      TEXT NOT NULL
  ground_truth  TEXT NOT NULL
  domain        VARCHAR(32) NOT NULL  -- tech | hr | org
  category      VARCHAR(64)           -- e.g. 'eks_incidents', 'leave_policy'
  source_doc_id VARCHAR(256)          -- Confluence page / Jira ticket
  created_at    TIMESTAMPTZ DEFAULT NOW()
  active        BOOLEAN DEFAULT TRUE
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from pipelines.components.shared.base import EnvConfig, get_logger
from src.ragas_eval.evaluator import RAGASInput

logger = get_logger(__name__)


# ── Schema management ─────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS golden_qa (
    id            SERIAL PRIMARY KEY,
    question      TEXT         NOT NULL,
    ground_truth  TEXT         NOT NULL,
    domain        VARCHAR(32)  NOT NULL,
    category      VARCHAR(64),
    source_doc_id VARCHAR(256),
    created_at    TIMESTAMPTZ  DEFAULT NOW(),
    active        BOOLEAN      DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_golden_qa_domain   ON golden_qa(domain);
CREATE INDEX IF NOT EXISTS idx_golden_qa_active   ON golden_qa(active);
CREATE INDEX IF NOT EXISTS idx_golden_qa_category ON golden_qa(category);
"""


@dataclass
class QAPair:
    question: str
    ground_truth: str
    domain: str
    category: str = ""
    source_doc_id: str = ""
    id: int = -1


# ── DB connection ─────────────────────────────────────────────────────────────

def _get_conn(cfg: EnvConfig):
    return psycopg2.connect(
        host=cfg.postgres_host,
        dbname=cfg.postgres_db,
        user=cfg.postgres_user,
        password=cfg.postgres_password,
    )


def ensure_schema(cfg: EnvConfig | None = None) -> None:
    cfg = cfg or EnvConfig()
    with _get_conn(cfg) as conn, conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    logger.info("golden_qa schema ensured")


# ── CRUD ──────────────────────────────────────────────────────────────────────

def insert_pairs(
    pairs: list[QAPair],
    cfg: EnvConfig | None = None,
) -> int:
    """
    Bulk-insert QA pairs.  Skips exact duplicates (question + domain).
    Returns number of rows actually inserted.
    """
    cfg = cfg or EnvConfig()
    inserted = 0
    with _get_conn(cfg) as conn, conn.cursor() as cur:
        for p in pairs:
            cur.execute(
                """
                INSERT INTO golden_qa (question, ground_truth, domain, category, source_doc_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (p.question, p.ground_truth, p.domain, p.category, p.source_doc_id),
            )
            inserted += cur.rowcount
        conn.commit()
    logger.info("Inserted %d / %d QA pairs", inserted, len(pairs))
    return inserted


def load_pairs(
    domain: str | None = None,
    category: str | None = None,
    limit: int = 200,
    cfg: EnvConfig | None = None,
) -> list[QAPair]:
    """
    Load active QA pairs, optionally filtered by domain / category.
    """
    cfg = cfg or EnvConfig()
    filters, params = ["active = TRUE"], []
    if domain:
        filters.append("domain = %s")
        params.append(domain)
    if category:
        filters.append("category = %s")
        params.append(category)
    params.append(limit)

    sql = f"SELECT id, question, ground_truth, domain, category, source_doc_id FROM golden_qa WHERE {' AND '.join(filters)} ORDER BY id LIMIT %s"

    with _get_conn(cfg) as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    pairs = [
        QAPair(
            id=r["id"],
            question=r["question"],
            ground_truth=r["ground_truth"],
            domain=r["domain"],
            category=r["category"] or "",
            source_doc_id=r["source_doc_id"] or "",
        )
        for r in rows
    ]
    logger.info("Loaded %d QA pairs (domain=%s, category=%s)", len(pairs), domain, category)
    return pairs


def coverage_report(cfg: EnvConfig | None = None) -> dict[str, Any]:
    """
    Returns per-domain, per-category pair counts.
    Used by Grafana dashboard 02 QA coverage heatmap.
    """
    cfg = cfg or EnvConfig()
    with _get_conn(cfg) as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            SELECT domain, category, COUNT(*) as count
            FROM golden_qa
            WHERE active = TRUE
            GROUP BY domain, category
            ORDER BY domain, count DESC
            """
        )
        rows = cur.fetchall()

    report: dict[str, Any] = {}
    for row in rows:
        d, cat, cnt = row["domain"], row["category"], row["count"]
        report.setdefault(d, {})
        report[d][cat or "_uncategorised"] = cnt

    return report


# ── Seed from JSON files ──────────────────────────────────────────────────────

def seed_from_json(json_path: str | Path, cfg: EnvConfig | None = None) -> int:
    """
    Load QA pairs from a JSON file and insert into PostgreSQL.

    Expected JSON format:
    [
      {
        "question": "Why does the payment service timeout?",
        "ground_truth": "Pod evictions caused by memory pressure...",
        "domain": "tech",
        "category": "eks_incidents",
        "source_doc_id": "JIRA-1234"
      },
      ...
    ]
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"QA dataset file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    pairs = [
        QAPair(
            question=item["question"],
            ground_truth=item["ground_truth"],
            domain=item["domain"],
            category=item.get("category", ""),
            source_doc_id=item.get("source_doc_id", ""),
        )
        for item in raw
    ]
    return insert_pairs(pairs, cfg=cfg)


# ── Convert to RAGAS inputs ───────────────────────────────────────────────────

def pairs_to_ragas_inputs(
    pairs: list[QAPair],
    answers: list[str],
    contexts_list: list[list[str]],
) -> list[RAGASInput]:
    """
    Combine QA pairs with generated answers and retrieved contexts
    into RAGASInput objects ready for evaluation.

    Args:
        pairs:          QA pairs from the golden dataset.
        answers:        LLM-generated answers (same order as pairs).
        contexts_list:  Retrieved context chunks per question (same order).

    Returns:
        List of RAGASInput ready for run_ragas_evaluation().
    """
    if not (len(pairs) == len(answers) == len(contexts_list)):
        raise ValueError(
            f"Length mismatch: pairs={len(pairs)}, "
            f"answers={len(answers)}, contexts={len(contexts_list)}"
        )

    return [
        RAGASInput(
            question=pair.question,
            answer=answer,
            contexts=contexts,
            ground_truth=pair.ground_truth,
            domain=pair.domain,
        )
        for pair, answer, contexts in zip(pairs, answers, contexts_list)
    ]


def load_golden_qa_from_files(data_dir: str) -> list[QAPair]:
    """
    Load all golden QA pairs from JSON files on disk.
    Used by dvc.yaml ragas_baseline stage and __main__.py.

    Args:
        data_dir: directory containing tech/, hr/, org/ subdirs with qa_pairs.json

    Returns:
        List of QAPair objects from all domains.
    """
    import json
    from pathlib import Path

    data_path = Path(data_dir)
    all_pairs: list[QAPair] = []

    for domain in ["tech", "hr", "org"]:
        qa_file = data_path / domain / "qa_pairs.json"
        if not qa_file.exists():
            continue
        raw = json.loads(qa_file.read_text(encoding="utf-8"))
        for item in raw:
            pair = QAPair(
                question=item["question"],
                ground_truth=item["ground_truth"],
                domain=item.get("domain", domain),
                category=item.get("category", ""),
                source_doc_id=item.get("source_doc_id", ""),
            )
            all_pairs.append(pair)

    return all_pairs
