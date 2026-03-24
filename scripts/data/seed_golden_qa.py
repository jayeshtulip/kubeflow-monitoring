import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS golden_qa ("
    "    id SERIAL PRIMARY KEY,"
    "    question TEXT NOT NULL,"
    "    ground_truth TEXT NOT NULL,"
    "    domain VARCHAR(32) NOT NULL,"
    "    category VARCHAR(64),"
    "    source_doc_id VARCHAR(256),"
    "    created_at TIMESTAMP DEFAULT NOW(),"
    "    active BOOLEAN DEFAULT TRUE)"
)

def seed(domain_filter="", clear=False):
    import psycopg2
    host     = os.environ.get("POSTGRES_HOST", "localhost")
    dbname   = os.environ.get("POSTGRES_DB", "llm_platform")
    user     = os.environ.get("POSTGRES_USER", "llm_admin")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    port     = int(os.environ.get("POSTGRES_PORT", "5432"))
    conn = psycopg2.connect(host=host,dbname=dbname,user=user,password=password,port=port)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(CREATE_SQL)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gqa_domain ON golden_qa(domain)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gqa_active ON golden_qa(active)")
        conn.commit()
        if clear:
            cur.execute("DELETE FROM golden_qa"); conn.commit()
            print("Cleared golden_qa.")
        data_dir = os.path.join(PROJECT_ROOT, "data", "golden_qa")
        domains = [domain_filter] if domain_filter else ["tech","hr","org"]
        tot_ins = tot_skip = 0
        for domain in domains:
            fpath = os.path.join(data_dir, domain, "qa_pairs.json")
            if not os.path.exists(fpath): print(f"SKIP {fpath}"); continue
            with open(fpath, encoding="utf-8") as f: pairs = json.load(f)
            ins = skip = 0
            for pair in pairs:
                cur.execute("SELECT id FROM golden_qa WHERE question=%s AND domain=%s",(pair["question"],pair["domain"]))
                if cur.fetchone(): skip += 1; continue
                cur.execute("INSERT INTO golden_qa (question,ground_truth,domain,category,source_doc_id) VALUES (%s,%s,%s,%s,%s)",(pair["question"],pair["ground_truth"],pair["domain"],pair.get("category",""),pair.get("source_doc_id","")))
                ins += 1
            conn.commit()
            print(f"Domain {domain}: {ins} inserted, {skip} skipped")
            tot_ins += ins; tot_skip += skip
    conn.close()
    print(f"Total: {tot_ins} inserted, {tot_skip} skipped")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="")
    p.add_argument("--clear", action="store_true")
    a = p.parse_args()
    seed(domain_filter=a.domain, clear=a.clear)