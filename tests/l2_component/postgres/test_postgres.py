"""L2 Component tests — PostgreSQL models and schema."""
import pytest


@pytest.mark.l2
@pytest.mark.timeout(10)
def test_postgres_connection(postgres_conn):
    """Assert PostgreSQL connection is healthy."""
    with postgres_conn.cursor() as cur:
        cur.execute("SELECT version()")
        row = cur.fetchone()
    assert row is not None
    assert "PostgreSQL" in str(row[0])


@pytest.mark.l2
@pytest.mark.timeout(15)
def test_golden_qa_table_exists(postgres_conn):
    """Assert golden_qa table has correct columns."""
    with postgres_conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s
        """, ("golden_qa",))
        cols = {row[0] for row in cur.fetchall()}
    required = {"id", "question", "ground_truth", "domain", "active"}
    missing = required - cols
    assert not missing, f"Missing columns in golden_qa: {missing}"


@pytest.mark.l2
@pytest.mark.timeout(15)
def test_golden_qa_has_minimum_coverage(postgres_conn):
    """Assert each domain has at least 20 active QA pairs."""
    with postgres_conn.cursor() as cur:
        cur.execute("""
            SELECT domain, COUNT(*) FROM golden_qa
            WHERE active = TRUE GROUP BY domain
        """)
        coverage = {row[0]: row[1] for row in cur.fetchall()}
    for domain in ["tech", "hr", "org"]:
        count = coverage.get(domain, 0)
        assert count >= 20, f"Domain {domain!r} has only {count} QA pairs (need >= 20)"


@pytest.mark.l2
@pytest.mark.timeout(15)
def test_sqlalchemy_models_create_tables(env_config):
    """SQLAlchemy models can create tables without error."""
    from src.storage.postgres.models import ensure_tables
    ensure_tables(env_config)