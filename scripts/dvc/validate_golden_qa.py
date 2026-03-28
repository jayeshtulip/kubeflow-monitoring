"""Validate golden QA coverage per domain from PostgreSQL."""
import json, os, sys
import psycopg2

host     = os.environ.get("POSTGRES_HOST", "llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com")
db       = os.environ.get("POSTGRES_DB", "llm_platform")
user     = os.environ.get("POSTGRES_USER", "llm_admin")
password = os.environ.get("POSTGRES_PASSWORD", "Llmplatform2026")

conn = psycopg2.connect(host=host, dbname=db, user=user, password=password, sslmode="require")
cur = conn.cursor()
cur.execute("SELECT domain, COUNT(*) FROM golden_qa WHERE active=TRUE GROUP BY domain")
rows = cur.fetchall()
conn.close()

coverage = {r[0]: r[1] for r in rows}
failures = [d for d in ["tech", "hr", "org"] if coverage.get(d, 0) < 10]

report = {"coverage": coverage, "failures": failures, "passed": len(failures) == 0}
print(json.dumps(report, indent=2))

os.makedirs("data/reports", exist_ok=True)
with open("data/reports/coverage_report.json", "w") as f:
    json.dump(report, f, indent=2)

if failures:
    print(f"FAILED: insufficient coverage for {failures}", file=sys.stderr)
    sys.exit(1)
print("PASSED: all domains have sufficient QA pairs")
