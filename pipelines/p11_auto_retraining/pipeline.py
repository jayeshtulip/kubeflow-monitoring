"""
P11 Auto-Retraining Pipeline - Enterprise LLM Platform v2.0
============================================================
Triggered by:
  1. P09 gate failure (faithfulness < 0.85)
  2. Evidently AI drift score exceeds threshold
  3. Nightly GitHub Actions schedule (Mon-Fri 03:30 UTC)

Pipeline stages:
  1. fetch_metrics_component     - pulls RAGAS history from MLflow + Prometheus
  2. evidently_drift_component   - runs Evidently AI drift detection
  3. update_golden_qa_component  - adds new QA pairs to PostgreSQL
  4. trigger_p10_component       - submits P10 DVC repro run (re-index Qdrant)
  5. trigger_p09_component       - submits P09 RAGAS eval run (verify improvement)
  6. log_improvement_component   - computes delta, logs to MLflow
"""

import kfp
from kfp import dsl
from kfp.dsl import Output, Artifact, Input


# ── Component 1: Fetch Metrics ─────────────────────────────────────────────

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=[
        "mlflow==2.11.1",
        "psycopg2-binary==2.9.9",
        "requests==2.31.0",
        "pandas==2.0.3",
        "protobuf==4.25.3",
    ],
)
def fetch_metrics_component(
    mlflow_tracking_uri: str,
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    prometheus_url: str,
    n_recent_runs: int,
    metrics_report: Output[Artifact],
) -> str:
    """
    Fetches:
      - Last N RAGAS runs from MLflow (faithfulness, context_precision, hallucination_rate)
      - Qdrant retrieval score distribution via Prometheus
      - vLLM inference latency P95 via Prometheus
      - Input question distribution stats from PostgreSQL golden_qa table
    """
    import json, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
                   capture_output=True)
    import mlflow
    import psycopg2
    import requests
    from datetime import datetime

    print(f"Fetching metrics | MLflow: {mlflow_tracking_uri} | n_runs: {n_recent_runs}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # 1. RAGAS scores from MLflow
    ragas_runs = []
    try:
        experiment = client.get_experiment_by_name("ragas-evaluation")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=n_recent_runs,
            )
            for run in runs:
                m = run.data.metrics
                ragas_runs.append({
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time,
                    "faithfulness": m.get("faithfulness", None),
                    "context_precision": m.get("context_precision", None),
                    "hallucination_rate": m.get("hallucination_rate", None),
                    "eval_seconds": m.get("eval_seconds", None),
                    "gate_passed": m.get("gate_passed", 0),
                })
            print(f"  RAGAS: fetched {len(ragas_runs)} runs")
        else:
            print("  WARNING: ragas-evaluation experiment not found")
    except Exception as e:
        print(f"  WARNING: MLflow RAGAS fetch failed: {e}")

    # 2. DVC pipeline runs from MLflow
    dvc_runs = []
    try:
        dvc_exp = client.get_experiment_by_name("dvc-pipeline")
        if dvc_exp:
            runs = client.search_runs(
                experiment_ids=[dvc_exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=5,
            )
            for run in runs:
                dvc_runs.append({
                    "run_id": run.info.run_id,
                    "dvc_sha": run.data.params.get("dvc_sha", ""),
                    "start_time": run.info.start_time,
                })
    except Exception as e:
        print(f"  WARNING: MLflow DVC fetch failed: {e}")

    # 3. Qdrant stats from Prometheus
    qdrant_stats = {}
    try:
        prom_base = prometheus_url.rstrip("/")
        for collection in ["tech_docs", "hr_policies", "org_info"]:
            query = f'qdrant_collection_vector_count{{collection="{collection}"}}'
            resp = requests.get(f"{prom_base}/api/v1/query", params={"query": query}, timeout=10)
            if resp.ok:
                result = resp.json().get("data", {}).get("result", [])
                if result:
                    qdrant_stats[f"{collection}_count"] = int(float(result[0]["value"][1]))
        print(f"  Qdrant stats: {qdrant_stats}")
    except Exception as e:
        print(f"  WARNING: Prometheus Qdrant query failed: {e}")

    # 4. vLLM latency from Prometheus
    vllm_stats = {}
    try:
        prom_base = prometheus_url.rstrip("/")
        query = 'histogram_quantile(0.95, rate(vllm_request_latency_seconds_bucket[1h]))'
        resp = requests.get(f"{prom_base}/api/v1/query", params={"query": query}, timeout=10)
        if resp.ok:
            result = resp.json().get("data", {}).get("result", [])
            vllm_stats["p95_latency_s"] = float(result[0]["value"][1]) if result else None
        query = 'rate(vllm_generation_tokens_total[5m])'
        resp = requests.get(f"{prom_base}/api/v1/query", params={"query": query}, timeout=10)
        if resp.ok:
            result = resp.json().get("data", {}).get("result", [])
            vllm_stats["tokens_per_second"] = float(result[0]["value"][1]) if result else None
        print(f"  vLLM stats: {vllm_stats}")
    except Exception as e:
        print(f"  WARNING: Prometheus vLLM query failed: {e}")
        vllm_stats = {"p95_latency_s": None, "tokens_per_second": None}

    # 5. Question distribution from PostgreSQL
    question_stats = {}
    try:
        conn = psycopg2.connect(
            host=postgres_host, database=postgres_db,
            user=postgres_user, password=postgres_password,
            port=5432, sslmode="require",
        )
        cur = conn.cursor()
        cur.execute("SELECT domain, COUNT(*) FROM golden_qa WHERE active=TRUE GROUP BY domain ORDER BY domain")
        domain_counts = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute("SELECT MAX(created_at) FROM golden_qa WHERE active=TRUE")
        row = cur.fetchone()
        question_stats = {
            "domain_counts": domain_counts,
            "latest_qa_added": str(row[0]) if row and row[0] else None,
            "total_active": sum(domain_counts.values()),
        }
        cur.close()
        conn.close()
        print(f"  Golden QA: {question_stats}")
    except Exception as e:
        print(f"  WARNING: PostgreSQL question stats failed: {e}")
        question_stats = {"domain_counts": {}, "total_active": 0}

    # Assemble report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "ragas_runs": ragas_runs,
        "dvc_runs": dvc_runs,
        "qdrant_stats": qdrant_stats,
        "vllm_stats": vllm_stats,
        "question_stats": question_stats,
        "n_ragas_runs_fetched": len(ragas_runs),
    }

    import pathlib
    pathlib.Path(metrics_report.path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_report.path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nMetrics report saved: {len(ragas_runs)} RAGAS runs")
    return f"fetched:{len(ragas_runs)}_runs"


# ── Component 2: Evidently Drift Detection ─────────────────────────────────

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=[
        "evidently==0.4.16",
        "pandas==2.0.3",
        "scipy==1.11.4",
        "protobuf==4.25.3",
    ],
)
def evidently_drift_component(
    metrics_report: Input[Artifact],
    mlflow_tracking_uri: str,
    faithfulness_drift_threshold: float,
    context_precision_drift_threshold: float,
    latency_drift_threshold_s: float,
    drift_report: Output[Artifact],
) -> str:
    """
    Runs Evidently AI drift detection on RAGAS metrics:
      - DataDriftPreset on faithfulness, context_precision, hallucination_rate
      - Jensen-Shannon divergence statistical test
      - vLLM P95 latency check vs baseline
      - Question domain imbalance check
    Returns: 'retrain' | 'no_action' | 'insufficient_data'
    """
    import json, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
                   capture_output=True)
    import pandas as pd
    import pathlib
    from datetime import datetime

    with open(metrics_report.path) as f:
        metrics = json.load(f)

    ragas_runs = metrics.get("ragas_runs", [])
    vllm_stats = metrics.get("vllm_stats", {})
    question_stats = metrics.get("question_stats", {})

    print(f"Evidently drift analysis | {len(ragas_runs)} RAGAS runs available")

    drift_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_runs_analyzed": len(ragas_runs),
        "drift_detected": False,
        "drift_reasons": [],
        "metric_details": {},
        "recommendation": "no_action",
    }

    if len(ragas_runs) < 2:
        print("  INSUFFICIENT DATA: need >=2 RAGAS runs for drift detection")
        drift_results["recommendation"] = "insufficient_data"
        pathlib.Path(drift_report.path).parent.mkdir(parents=True, exist_ok=True)
        with open(drift_report.path, "w") as f:
            json.dump(drift_results, f, indent=2)
        return "insufficient_data"

    # Build DataFrame
    df = pd.DataFrame([r for r in ragas_runs if r.get("faithfulness") is not None])
    df = df.sort_values("start_time").reset_index(drop=True)

    if len(df) >= 4:
        mid = len(df) // 2
        ref_df = df.iloc[:mid]
        cur_df = df.iloc[mid:]
    else:
        ref_df = df.iloc[:-1]
        cur_df = df.iloc[-1:]

    print(f"  Reference runs: {len(ref_df)} | Current runs: {len(cur_df)}")

    # Evidently AI DataDrift Report
    dataset_drift = False
    try:
        from evidently.report import Report
        from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

        feature_cols = ["faithfulness", "context_precision", "hallucination_rate"]
        ref_ev = ref_df[feature_cols].dropna()
        cur_ev = cur_df[feature_cols].dropna()

        report = Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name="faithfulness"),
            ColumnDriftMetric(column_name="context_precision"),
            ColumnDriftMetric(column_name="hallucination_rate"),
        ])
        report.run(reference_data=ref_ev, current_data=cur_ev)
        rd = report.as_dict()

        dataset_drift = rd["metrics"][0]["result"].get("dataset_drift", False)
        share_drifted = rd["metrics"][0]["result"].get("share_of_drifted_columns", 0.0)
        drift_results["evidently_dataset_drift"] = dataset_drift
        drift_results["evidently_share_drifted_columns"] = share_drifted

        for i, col in enumerate(feature_cols, start=1):
            col_result = rd["metrics"][i]["result"]
            col_drifted = col_result.get("drift_detected", False)
            drift_score = col_result.get("drift_score", 0.0)
            drift_results["metric_details"][col] = {
                "drift_detected": col_drifted,
                "drift_score": round(drift_score, 4),
                "stattest": col_result.get("stattest_name", ""),
            }
            if col_drifted:
                drift_results["drift_reasons"].append(
                    f"Evidently: {col} drift_score={drift_score:.4f}"
                )
        print(f"  Evidently: dataset_drift={dataset_drift}, share_drifted={share_drifted:.2f}")
    except Exception as e:
        print(f"  WARNING: Evidently report failed: {e}")

    # Manual threshold checks
    # 1. Faithfulness delta
    if len(cur_df) > 0 and "faithfulness" in cur_df.columns:
        cur_faith = cur_df["faithfulness"].dropna().mean()
        ref_faith = ref_df["faithfulness"].dropna().mean()
        faith_delta = cur_faith - ref_faith
        drift_results["metric_details"]["faithfulness_current_mean"] = round(cur_faith, 4)
        drift_results["metric_details"]["faithfulness_ref_mean"] = round(ref_faith, 4)
        drift_results["metric_details"]["faithfulness_delta"] = round(faith_delta, 4)
        if abs(faith_delta) > faithfulness_drift_threshold:
            drift_results["drift_reasons"].append(
                f"Faithfulness delta {faith_delta:.4f} exceeds threshold {faithfulness_drift_threshold}"
            )
            drift_results["drift_detected"] = True
        if cur_faith < 0.60:
            drift_results["drift_reasons"].append(
                f"Faithfulness {cur_faith:.4f} below critical floor 0.60"
            )
            drift_results["drift_detected"] = True
        print(f"  Faithfulness: ref={ref_faith:.4f} cur={cur_faith:.4f} delta={faith_delta:.4f}")

    # 2. Context precision delta
    if "context_precision" in cur_df.columns:
        cur_cp = cur_df["context_precision"].dropna().mean()
        ref_cp = ref_df["context_precision"].dropna().mean()
        cp_delta = cur_cp - ref_cp
        drift_results["metric_details"]["context_precision_delta"] = round(cp_delta, 4)
        if abs(cp_delta) > context_precision_drift_threshold:
            drift_results["drift_reasons"].append(
                f"Context precision delta {cp_delta:.4f} exceeds threshold {context_precision_drift_threshold}"
            )
            drift_results["drift_detected"] = True
        print(f"  Context precision: ref={ref_cp:.4f} cur={cur_cp:.4f} delta={cp_delta:.4f}")

    # 3. vLLM latency
    p95_latency = vllm_stats.get("p95_latency_s")
    drift_results["metric_details"]["vllm_p95_latency_s"] = p95_latency
    if p95_latency is not None and p95_latency > latency_drift_threshold_s:
        drift_results["drift_reasons"].append(
            f"vLLM P95 latency {p95_latency:.2f}s exceeds threshold {latency_drift_threshold_s}s"
        )
        drift_results["drift_detected"] = True
        print(f"  vLLM latency DRIFT: {p95_latency:.2f}s > {latency_drift_threshold_s}s")
    else:
        print(f"  vLLM latency: {p95_latency}s (OK)")

    # 4. Domain imbalance
    domain_counts = question_stats.get("domain_counts", {})
    if domain_counts:
        counts = list(domain_counts.values())
        if max(counts) > 0:
            imbalance_ratio = min(counts) / max(counts)
            drift_results["metric_details"]["question_domain_imbalance"] = round(imbalance_ratio, 4)
            if imbalance_ratio < 0.5:
                drift_results["drift_reasons"].append(
                    f"Question domain imbalance ratio={imbalance_ratio:.4f}"
                )
                drift_results["drift_detected"] = True
        print(f"  Domain counts: {domain_counts}")

    if dataset_drift:
        drift_results["drift_detected"] = True
        drift_results["drift_reasons"].append("Evidently dataset-level drift detected")

    drift_results["recommendation"] = "retrain" if drift_results["drift_detected"] else "no_action"

    if drift_results["drift_detected"]:
        print(f"\n  DRIFT DETECTED: {drift_results['drift_reasons']}")
    else:
        print(f"\n  NO DRIFT DETECTED - system within tolerance")

    pathlib.Path(drift_report.path).parent.mkdir(parents=True, exist_ok=True)
    with open(drift_report.path, "w") as f:
        json.dump(drift_results, f, indent=2)

    return drift_results["recommendation"]


# ── Component 3: Update Golden QA ─────────────────────────────────────────

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=[
        "psycopg2-binary==2.9.9",
        "protobuf==4.25.3",
    ],
)
def update_golden_qa_component(
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    qdrant_host: str,
    drift_report: Input[Artifact],
    qa_update_report: Output[Artifact],
) -> int:
    """
    Inserts new golden QA pairs into PostgreSQL for domains below 25 pairs.
    QA pairs are based on the new P10 documents (eks_gpu_setup, vllm_performance_guide).
    Returns count of new QA pairs inserted.
    """
    import json, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
                   capture_output=True)
    import psycopg2
    import pathlib
    from datetime import datetime

    with open(drift_report.path) as f:
        drift = json.load(f)

    qa_inserted = 0
    domain_counts = {}
    update_report = {"timestamp": datetime.utcnow().isoformat(), "new_qa_pairs": [], "errors": []}

    try:
        conn = psycopg2.connect(
            host=postgres_host, database=postgres_db,
            user=postgres_user, password=postgres_password,
            port=5432, sslmode="require",
        )
        cur = conn.cursor()
        cur.execute("SELECT domain, COUNT(*) FROM golden_qa WHERE active=TRUE GROUP BY domain")
        domain_counts = {row[0]: row[1] for row in cur.fetchall()}
        print(f"Current QA coverage: {domain_counts}")

        new_qa_templates = {
            "tech": [
                {
                    "question": "What GPU instance type does the Enterprise LLM Platform use for vLLM inference?",
                    "ground_truth": "The platform uses g4dn.2xlarge instances with a Tesla T4 GPU (16GB VRAM) for vLLM inference.",
                    "source": "eks_gpu_setup.txt",
                },
                {
                    "question": "What is the token throughput achieved by vLLM with Mistral-7B-GPTQ on the T4 GPU?",
                    "ground_truth": "vLLM achieves 37 tokens per second with Mistral-7B-Instruct-v0.2-GPTQ (4-bit) on a Tesla T4 GPU.",
                    "source": "vllm_performance_guide.txt",
                },
                {
                    "question": "What CUDA version does the EKS GPU node use?",
                    "ground_truth": "The EKS GPU node uses CUDA 12.8, enabled by the AL2_x86_64_GPU AMI with pre-installed NVIDIA drivers.",
                    "source": "eks_gpu_setup.txt",
                },
                {
                    "question": "What percentage of T4 GPU SM utilization does vLLM achieve at peak?",
                    "ground_truth": "vLLM achieves 90% GPU SM utilization and 94% memory bandwidth utilization on the Tesla T4 at peak load.",
                    "source": "vllm_performance_guide.txt",
                },
                {
                    "question": "How much VRAM does the Mistral-7B-GPTQ model consume on the T4 GPU?",
                    "ground_truth": "vLLM serves TheBloke/Mistral-7B-Instruct-v0.2-GPTQ using GPTQ 4-bit quantization, consuming 12.9GB of the T4s 16GB VRAM.",
                    "source": "vllm_performance_guide.txt",
                },
            ],
        }

        for domain, templates in new_qa_templates.items():
            current_count = domain_counts.get(domain, 0)
            slots = max(0, 25 - current_count)
            if slots > 0:
                for qa in templates[:slots]:
                    cur.execute("""
                        INSERT INTO golden_qa (question, ground_truth, domain, source, active, created_at)
                        VALUES (%s, %s, %s, %s, TRUE, NOW())
                        ON CONFLICT (question) DO NOTHING
                        RETURNING id
                    """, (qa["question"], qa["ground_truth"], domain, qa.get("source", "")))
                    result = cur.fetchone()
                    if result:
                        qa_inserted += 1
                        update_report["new_qa_pairs"].append({
                            "domain": domain,
                            "question": qa["question"][:80] + "...",
                        })
                        print(f"  Inserted: [{domain}] {qa['question'][:60]}...")

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        update_report["errors"].append(str(e))
        print(f"  WARNING: Golden QA update failed: {e}")

    update_report["qa_inserted"] = qa_inserted
    update_report["domain_coverage_before"] = domain_counts
    print(f"\nGolden QA: {qa_inserted} new pairs inserted")

    pathlib.Path(qa_update_report.path).parent.mkdir(parents=True, exist_ok=True)
    with open(qa_update_report.path, "w") as f:
        json.dump(update_report, f, indent=2)
    return qa_inserted


# ── Component 4: Trigger P10 ───────────────────────────────────────────────

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["requests==2.31.0", "protobuf==4.25.3"],
)
def trigger_p10_component(
    kfp_endpoint: str,
    s3_remote: str,
    aws_region: str,
    qdrant_host: str,
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    mlflow_tracking_uri: str,
    run_uuid: str,
    p10_trigger_report: Output[Artifact],
) -> str:
    """
    Submits P10 DVC reproducibility pipeline via KFP REST API.
    Polls for completion (max 30 min). Returns final status.
    """
    import json, time, uuid, requests, pathlib
    from datetime import datetime

    trigger_uuid = f"p11-p10-{uuid.uuid4().hex[:8]}"
    print(f"Triggering P10 | uuid: {trigger_uuid}")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "trigger_uuid": trigger_uuid,
        "p10_run_id": None,
        "p10_status": "pending",
        "duration_s": 0,
        "error": None,
    }

    try:
        base = kfp_endpoint.rstrip("/")
        list_resp = requests.get(
            f'{base}/apis/v2beta1/pipelines?filter={{"predicates":[{{"key":"name","op":"EQUALS","string_value":"p10-dvc-reproducibility"}}]}}',
            timeout=10,
        )
        pipeline_id = None
        if list_resp.ok:
            items = list_resp.json().get("pipelines", [])
            if items:
                pipeline_id = items[0]["pipeline_id"]
                print(f"  Found P10 pipeline: {pipeline_id}")

        if not pipeline_id:
            report["error"] = "P10 pipeline not found in KFP registry"
            report["p10_status"] = "failed"
        else:
            run_payload = {
                "display_name": f"p10-p11-triggered-{trigger_uuid}",
                "pipeline_version_reference": {"pipeline_id": pipeline_id},
                "runtime_config": {
                    "parameters": {
                        "stage": {"string_value": "all"},
                        "s3_remote": {"string_value": s3_remote},
                        "aws_region": {"string_value": aws_region},
                        "qdrant_host": {"string_value": qdrant_host},
                        "postgres_host": {"string_value": postgres_host},
                        "postgres_db": {"string_value": postgres_db},
                        "postgres_user": {"string_value": postgres_user},
                        "postgres_password": {"string_value": postgres_password},
                        "mlflow_tracking_uri": {"string_value": mlflow_tracking_uri},
                        "run_uuid": {"string_value": trigger_uuid},
                    },
                    "enable_caching": False,
                },
            }
            run_resp = requests.post(f"{base}/apis/v2beta1/runs", json=run_payload, timeout=15)
            if run_resp.ok:
                run_id = run_resp.json()["run_id"]
                report["p10_run_id"] = run_id
                print(f"  P10 run submitted: {run_id}")
                max_wait, interval, elapsed = 1800, 30, 0
                while elapsed < max_wait:
                    time.sleep(interval)
                    elapsed += interval
                    sr = requests.get(f"{base}/apis/v2beta1/runs/{run_id}", timeout=10)
                    if sr.ok:
                        state = sr.json().get("state", "")
                        print(f"  P10 [{elapsed}s]: {state}")
                        if state in ("SUCCEEDED", "FAILED", "SKIPPED", "ERROR"):
                            report["p10_status"] = state.lower()
                            report["duration_s"] = elapsed
                            break
                else:
                    report["p10_status"] = "timeout"
            else:
                report["error"] = f"Submission failed: {run_resp.status_code}"
                report["p10_status"] = "failed"
    except Exception as e:
        report["error"] = str(e)
        report["p10_status"] = "failed"
        print(f"  ERROR: {e}")

    pathlib.Path(p10_trigger_report.path).parent.mkdir(parents=True, exist_ok=True)
    with open(p10_trigger_report.path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nP10 complete: status={report['p10_status']} duration={report['duration_s']}s")
    return report["p10_status"]


# ── Component 5: Trigger P09 ───────────────────────────────────────────────

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["requests==2.31.0", "mlflow==2.11.1", "protobuf==4.25.3"],
)
def trigger_p09_component(
    kfp_endpoint: str,
    mlflow_tracking_uri: str,
    ollama_base_url: str,
    vllm_base_url: str,
    qdrant_host: str,
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    run_uuid: str,
    p09_trigger_report: Output[Artifact],
) -> float:
    """
    Submits P09 RAGAS evaluation pipeline and returns new faithfulness score.
    Polls KFP for completion (max 60 min), then reads results from MLflow.
    """
    import json, time, uuid, requests, pathlib, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
                   capture_output=True)
    import mlflow
    from datetime import datetime

    trigger_uuid = f"p11-p09-{uuid.uuid4().hex[:8]}"
    print(f"Triggering P09 | uuid: {trigger_uuid}")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "trigger_uuid": trigger_uuid,
        "p09_run_id": None,
        "p09_status": "pending",
        "new_faithfulness": None,
        "new_context_precision": None,
        "new_hallucination_rate": None,
        "gate_passed": False,
        "error": None,
    }

    try:
        base = kfp_endpoint.rstrip("/")
        list_resp = requests.get(
            f'{base}/apis/v2beta1/pipelines?filter={{"predicates":[{{"key":"name","op":"EQUALS","string_value":"p09-ragas-evaluation"}}]}}',
            timeout=10,
        )
        pipeline_id = None
        if list_resp.ok:
            items = list_resp.json().get("pipelines", [])
            if items:
                pipeline_id = items[0]["pipeline_id"]
                print(f"  Found P09 pipeline: {pipeline_id}")

        if not pipeline_id:
            report["error"] = "P09 pipeline not found in KFP registry"
            report["p09_status"] = "failed"
        else:
            run_payload = {
                "display_name": f"p09-p11-triggered-{trigger_uuid}",
                "pipeline_version_reference": {"pipeline_id": pipeline_id},
                "runtime_config": {
                    "parameters": {
                        "qa_limit": {"int_value": 200},
                        "top_k": {"int_value": 5},
                        "embedding_model": {"string_value": "all-MiniLM-L6-v2"},
                        "ollama_base_url": {"string_value": ollama_base_url},
                        "vllm_base_url": {"string_value": vllm_base_url},
                        "qdrant_host": {"string_value": qdrant_host},
                        "postgres_host": {"string_value": postgres_host},
                        "postgres_db": {"string_value": postgres_db},
                        "postgres_user": {"string_value": postgres_user},
                        "postgres_password": {"string_value": postgres_password},
                        "mlflow_tracking_uri": {"string_value": mlflow_tracking_uri},
                        "faithfulness_hard_block": {"double_value": 0.85},
                        "hallucination_hard_block": {"double_value": 0.15},
                    },
                    "enable_caching": False,
                },
            }
            run_resp = requests.post(f"{base}/apis/v2beta1/runs", json=run_payload, timeout=15)
            if run_resp.ok:
                run_id = run_resp.json()["run_id"]
                report["p09_run_id"] = run_id
                print(f"  P09 run submitted: {run_id}")
                max_wait, interval, elapsed = 3600, 60, 0
                while elapsed < max_wait:
                    time.sleep(interval)
                    elapsed += interval
                    sr = requests.get(f"{base}/apis/v2beta1/runs/{run_id}", timeout=10)
                    if sr.ok:
                        state = sr.json().get("state", "")
                        print(f"  P09 [{elapsed}s]: {state}")
                        if state in ("SUCCEEDED", "FAILED", "SKIPPED", "ERROR"):
                            report["p09_status"] = state.lower()
                            report["duration_s"] = elapsed
                            break
                else:
                    report["p09_status"] = "timeout"

                if report["p09_status"] == "succeeded":
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                    client = mlflow.tracking.MlflowClient()
                    exp = client.get_experiment_by_name("ragas-evaluation")
                    if exp:
                        runs = client.search_runs(
                            experiment_ids=[exp.experiment_id],
                            order_by=["start_time DESC"],
                            max_results=1,
                        )
                        if runs:
                            m = runs[0].data.metrics
                            report["new_faithfulness"] = m.get("faithfulness")
                            report["new_context_precision"] = m.get("context_precision")
                            report["new_hallucination_rate"] = m.get("hallucination_rate")
                            report["gate_passed"] = bool(m.get("gate_passed", 0))
                            print(f"  New scores: faithfulness={report['new_faithfulness']}")
            else:
                report["error"] = f"Submission failed: {run_resp.status_code}"
                report["p09_status"] = "failed"
    except Exception as e:
        report["error"] = str(e)
        print(f"  ERROR: {e}")

    pathlib.Path(p09_trigger_report.path).parent.mkdir(parents=True, exist_ok=True)
    with open(p09_trigger_report.path, "w") as f:
        json.dump(report, f, indent=2)

    faithfulness = report.get("new_faithfulness") or 0.0
    print(f"\nP09 verification complete: faithfulness={faithfulness}")
    return faithfulness


# ── Component 6: Log Improvement ──────────────────────────────────────────

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["mlflow==2.11.1", "protobuf==4.25.3"],
)
def log_improvement_component(
    mlflow_tracking_uri: str,
    drift_report: Input[Artifact],
    metrics_report: Input[Artifact],
    p09_trigger_report: Input[Artifact],
    qa_update_report: Input[Artifact],
    run_uuid: str,
    improvement_report: Output[Artifact],
) -> str:
    """
    Computes improvement delta (pre vs post retraining) and logs to MLflow
    experiment 'p11-auto-retraining'. Returns outcome: PASS | IMPROVED | NO_CHANGE.
    """
    import json, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
                   capture_output=True)
    import mlflow
    import pathlib
    from datetime import datetime

    with open(drift_report.path) as f:
        drift = json.load(f)
    with open(metrics_report.path) as f:
        metrics = json.load(f)
    with open(p09_trigger_report.path) as f:
        p09 = json.load(f)
    with open(qa_update_report.path) as f:
        qa_update = json.load(f)

    ragas_runs = metrics.get("ragas_runs", [])
    pre_faith  = ragas_runs[0].get("faithfulness", 0.0) if ragas_runs else 0.0
    pre_ctx    = ragas_runs[0].get("context_precision", 0.0) if ragas_runs else 0.0
    pre_halluc = ragas_runs[0].get("hallucination_rate", 1.0) if ragas_runs else 1.0

    post_faith  = p09.get("new_faithfulness") or pre_faith
    post_ctx    = p09.get("new_context_precision") or pre_ctx
    post_halluc = p09.get("new_hallucination_rate") or pre_halluc

    faith_delta  = post_faith - pre_faith
    ctx_delta    = post_ctx - pre_ctx
    halluc_delta = post_halluc - pre_halluc
    gate_passed  = p09.get("gate_passed", False)
    improved     = faith_delta > 0.01 or gate_passed

    outcome = "PASS" if gate_passed else ("IMPROVED" if improved else "NO_CHANGE")

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_uuid": run_uuid,
        "drift_detected": drift.get("drift_detected", False),
        "drift_reasons": drift.get("drift_reasons", []),
        "pre_retraining": {"faithfulness": round(pre_faith, 4), "context_precision": round(pre_ctx, 4), "hallucination_rate": round(pre_halluc, 4)},
        "post_retraining": {"faithfulness": round(post_faith, 4), "context_precision": round(post_ctx, 4), "hallucination_rate": round(post_halluc, 4)},
        "improvement_delta": {"faithfulness": round(faith_delta, 4), "context_precision": round(ctx_delta, 4), "hallucination_rate": round(halluc_delta, 4)},
        "gate_passed": gate_passed,
        "new_qa_pairs_added": qa_update.get("qa_inserted", 0),
        "p09_run_id": p09.get("p09_run_id"),
        "outcome": outcome,
    }

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("p11-auto-retraining")
        with mlflow.start_run(run_name=f"p11-{run_uuid[:8]}"):
            mlflow.log_param("run_uuid", run_uuid)
            mlflow.log_param("drift_detected", drift.get("drift_detected"))
            mlflow.log_param("qa_pairs_added", qa_update.get("qa_inserted", 0))
            mlflow.log_metric("pre_faithfulness", pre_faith)
            mlflow.log_metric("post_faithfulness", post_faith)
            mlflow.log_metric("faithfulness_delta", faith_delta)
            mlflow.log_metric("pre_context_precision", pre_ctx)
            mlflow.log_metric("post_context_precision", post_ctx)
            mlflow.log_metric("context_precision_delta", ctx_delta)
            mlflow.log_metric("pre_hallucination_rate", pre_halluc)
            mlflow.log_metric("post_hallucination_rate", post_halluc)
            mlflow.log_metric("hallucination_delta", halluc_delta)
            mlflow.log_metric("gate_passed", int(gate_passed))
            mlflow.set_tag("pipeline", "P11-AUTO-RETRAIN")
            mlflow.set_tag("outcome", outcome)
        print(f"  MLflow: logged to p11-auto-retraining")
    except Exception as e:
        print(f"  WARNING: MLflow logging failed: {e}")

    print(f"\n{'='*55}")
    print(f"P11 RETRAINING SUMMARY")
    print(f"{'='*55}")
    print(f"  Drift detected:  {summary['drift_detected']}")
    print(f"  Faithfulness:    {pre_faith:.4f} -> {post_faith:.4f} (d{faith_delta:+.4f})")
    print(f"  Ctx Precision:   {pre_ctx:.4f} -> {post_ctx:.4f} (d{ctx_delta:+.4f})")
    print(f"  Hallucination:   {pre_halluc:.4f} -> {post_halluc:.4f} (d{halluc_delta:+.4f})")
    print(f"  Gate passed:     {gate_passed}")
    print(f"  QA pairs added:  {qa_update.get('qa_inserted', 0)}")
    print(f"  Outcome:         {outcome}")
    print(f"{'='*55}")

    pathlib.Path(improvement_report.path).parent.mkdir(parents=True, exist_ok=True)
    with open(improvement_report.path, "w") as f:
        json.dump(summary, f, indent=2)
    return outcome


# ── Pipeline Definition ────────────────────────────────────────────────────

@dsl.pipeline(
    name="p11-auto-retraining",
    description="Evidently AI drift detection + auto-retraining pipeline for Enterprise LLM Platform v2.0",
)
def p11_auto_retraining_pipeline(
    mlflow_tracking_uri: str = "http://172.20.172.203:5000",
    kfp_endpoint: str = "http://ml-pipeline.kubeflow.svc.cluster.local:8888",
    prometheus_url: str = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
    postgres_host: str = "llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com",
    postgres_db: str = "llm_platform",
    postgres_user: str = "llm_admin",
    postgres_password: str = "Llmplatform2026",
    qdrant_host: str = "qdrant.llm-platform-prod.svc.cluster.local",
    vllm_base_url: str = "http://vllm-service.llm-platform-prod.svc.cluster.local:8000",
    ollama_base_url: str = "http://ollama-service.llm-platform-prod.svc.cluster.local:11434",
    s3_remote: str = "s3://llm-platform-dvc-remote-659071697671/dvc",
    aws_region: str = "us-east-1",
    faithfulness_drift_threshold: float = 0.05,
    context_precision_drift_threshold: float = 0.05,
    latency_drift_threshold_s: float = 5.0,
    n_recent_runs: int = 10,
    run_uuid: str = "p11-default",
    force_retrain: bool = False,
):
    fetch_task = fetch_metrics_component(
        mlflow_tracking_uri=mlflow_tracking_uri,
        postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        prometheus_url=prometheus_url, n_recent_runs=n_recent_runs,
    )
    fetch_task.set_caching_options(False)

    drift_task = evidently_drift_component(
        metrics_report=fetch_task.outputs["metrics_report"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        faithfulness_drift_threshold=faithfulness_drift_threshold,
        context_precision_drift_threshold=context_precision_drift_threshold,
        latency_drift_threshold_s=latency_drift_threshold_s,
    )
    drift_task.set_caching_options(False)

    qa_task = update_golden_qa_component(
        postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        qdrant_host=qdrant_host,
        drift_report=drift_task.outputs["drift_report"],
    )
    qa_task.set_caching_options(False)

    p10_task = trigger_p10_component(
        kfp_endpoint=kfp_endpoint, s3_remote=s3_remote, aws_region=aws_region,
        qdrant_host=qdrant_host, postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        mlflow_tracking_uri=mlflow_tracking_uri, run_uuid=run_uuid,
    )
    p10_task.set_caching_options(False)
    p10_task.after(qa_task)

    p09_task = trigger_p09_component(
        kfp_endpoint=kfp_endpoint, mlflow_tracking_uri=mlflow_tracking_uri,
        ollama_base_url=ollama_base_url, vllm_base_url=vllm_base_url,
        qdrant_host=qdrant_host, postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        run_uuid=run_uuid,
    )
    p09_task.set_caching_options(False)
    p09_task.after(p10_task)

    log_task = log_improvement_component(
        mlflow_tracking_uri=mlflow_tracking_uri,
        drift_report=drift_task.outputs["drift_report"],
        metrics_report=fetch_task.outputs["metrics_report"],
        p09_trigger_report=p09_task.outputs["p09_trigger_report"],
        qa_update_report=qa_task.outputs["qa_update_report"],
        run_uuid=run_uuid,
    )
    log_task.set_caching_options(False)


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=p11_auto_retraining_pipeline,
        package_path="p11_pipeline.yaml",
    )
    print("Compiled: p11_pipeline.yaml")

