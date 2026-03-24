"""
DVC metrics Prometheus exporter.
Polls DVC + S3 and pushes to Prometheus:
  - last_commit_time     : Unix timestamp of most recent DVC data commit
  - dataset_sizes        : Vector count per Qdrant collection
  - embedding_drift_score: JS-divergence of embeddings (from Evidently)
  - qa_coverage          : Active golden QA pairs per domain

Runs as a sidecar pod in the monitoring namespace.
Scraped by Prometheus every 60s.
"""
from __future__ import annotations
import time
import subprocess
from prometheus_client import start_http_server, Gauge, CollectorRegistry
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)

EXPORTER_REGISTRY = CollectorRegistry()

DVC_LAST_COMMIT_TIME = Gauge(
    "dvc_last_commit_timestamp_seconds",
    "Unix timestamp of most recent DVC data commit",
    registry=EXPORTER_REGISTRY,
)

QDRANT_VECTOR_COUNT = Gauge(
    "qdrant_collection_vector_count",
    "Number of indexed vectors per collection",
    ["collection"],
    registry=EXPORTER_REGISTRY,
)

EMBEDDING_DRIFT_SCORE = Gauge(
    "embedding_drift_js_divergence",
    "Jensen-Shannon divergence of query embedding distributions",
    registry=EXPORTER_REGISTRY,
)

QA_COVERAGE = Gauge(
    "golden_qa_pair_count",
    "Active golden QA pairs per domain",
    ["domain"],
    registry=EXPORTER_REGISTRY,
)


def collect_dvc_last_commit() -> float:
    """Get Unix timestamp of most recent DVC commit via git log."""
    try:
        result = subprocess.run(
            ["git", "log", "--format=%ct", "-1", "--", "*.dvc"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as exc:
        logger.error("DVC commit time error: %s", exc)
    return 0.0


def collect_qdrant_stats(cfg: EnvConfig) -> dict[str, int]:
    """Get vector count per collection from Qdrant."""
    try:
        from qdrant_client import QdrantClient
        from pipelines.components.shared.base import get_qdrant_client
        client = get_qdrant_client(cfg)
        result = {}
        for col in client.get_collections().collections:
            info = client.get_collection(col.name)
            result[col.name] = info.vectors_count or 0
        return result
    except Exception as exc:
        logger.error("Qdrant stats error: %s", exc)
        return {}


def collect_qa_coverage(cfg: EnvConfig) -> dict[str, int]:
    """Get active QA pair count per domain from PostgreSQL."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=cfg.postgres_host, dbname=cfg.postgres_db,
            user=cfg.postgres_user, password=cfg.postgres_password,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT domain, COUNT(*) FROM golden_qa"
                " WHERE active=TRUE GROUP BY domain"
            )
            result = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
        return result
    except Exception as exc:
        logger.error("QA coverage error: %s", exc)
        return {}


def run_exporter(port: int = 9101, scrape_interval: int = 60) -> None:
    """Start Prometheus HTTP server and collect metrics in a loop."""
    cfg = EnvConfig()
    start_http_server(port, registry=EXPORTER_REGISTRY)
    logger.info("DVC Prometheus exporter started on port %d", port)
    while True:
        try:
            ts = collect_dvc_last_commit()
            DVC_LAST_COMMIT_TIME.set(ts)
            for col, count in collect_qdrant_stats(cfg).items():
                QDRANT_VECTOR_COUNT.labels(collection=col).set(count)
            for domain, count in collect_qa_coverage(cfg).items():
                QA_COVERAGE.labels(domain=domain).set(count)
            logger.info("Exporter metrics updated")
        except Exception as exc:
            logger.error("Exporter collect error: %s", exc)
        time.sleep(scrape_interval)


if __name__ == "__main__":
    run_exporter()