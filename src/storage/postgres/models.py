"""
SQLAlchemy models for PostgreSQL.
Tables:
  - golden_qa       : RAGAS golden question-answer pairs
  - ragas_results   : RAGAS evaluation run results
  - audit_log       : API request audit trail
  - platform_health : P04 quality monitoring snapshots
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    DateTime, JSON, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class GoldenQA(Base):
    __tablename__ = "golden_qa"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    question      = Column(Text, nullable=False)
    ground_truth  = Column(Text, nullable=False)
    domain        = Column(String(32), nullable=False, index=True)
    category      = Column(String(64), index=True)
    source_doc_id = Column(String(256))
    created_at    = Column(DateTime, default=datetime.utcnow)
    active        = Column(Boolean, default=True, index=True)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "question":     self.question,
            "ground_truth": self.ground_truth,
            "domain":       self.domain,
            "category":     self.category,
            "source_doc_id": self.source_doc_id,
        }


class RAGASResult(Base):
    __tablename__ = "ragas_results"
    id                  = Column(Integer, primary_key=True, autoincrement=True)
    mlflow_run_id       = Column(String(64), nullable=False, unique=True)
    faithfulness        = Column(Float)
    answer_relevancy    = Column(Float)
    context_precision   = Column(Float)
    context_recall      = Column(Float)
    answer_correctness  = Column(Float)
    hallucination_rate  = Column(Float)
    sample_count        = Column(Integer)
    gate_passed         = Column(Boolean)
    gate_failures       = Column(JSON)
    eval_seconds        = Column(Float)
    pipeline            = Column(String(32), default="P09")
    created_at          = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String(64), index=True)
    query_hash    = Column(String(64), index=True)
    workflow_used = Column(String(32))
    latency_ms    = Column(Float)
    tool_calls    = Column(Integer)
    replan_count  = Column(Integer)
    guardrail_hit = Column(Boolean, default=False)
    success       = Column(Boolean)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)


class PlatformHealth(Base):
    __tablename__ = "platform_health"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    status             = Column(String(16), index=True)
    health_score       = Column(Float)
    p95_latency_ms     = Column(Float)
    success_rate       = Column(Float)
    hallucination_rate = Column(Float)
    violations         = Column(JSON)
    mlflow_run_id      = Column(String(64))
    created_at         = Column(DateTime, default=datetime.utcnow, index=True)


# ── Engine factory ────────────────────────────────────────────────────────

def get_engine(cfg: EnvConfig | None = None):
    cfg = cfg or EnvConfig()
    url = (
        f"postgresql+psycopg2://{cfg.postgres_user}:{cfg.postgres_password}"
        f"@{cfg.postgres_host}/{cfg.postgres_db}"
    )
    return create_engine(url, pool_size=5, max_overflow=10, pool_pre_ping=True)


def ensure_tables(cfg: EnvConfig | None = None) -> None:
    """Create all tables if they do not exist."""
    engine = get_engine(cfg)
    Base.metadata.create_all(engine)
    logger.info("PostgreSQL tables ensured")


def get_session(cfg: EnvConfig | None = None) -> Session:
    from sqlalchemy.orm import sessionmaker
    engine = get_engine(cfg)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()