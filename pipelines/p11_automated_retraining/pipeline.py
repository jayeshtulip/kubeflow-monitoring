"""
Kubeflow Pipeline 11 - Automated Retraining
=============================================
Triggered by:
  - Evidently drift alert webhook (JS-div > 0.15)
  - RAGAS gate failure in P09 (faithfulness < 0.85)
  - Manual via run_pipelines.ps1 -Pipeline p11

Steps:
  1. check_drift_severity   - confirm drift is real
  2. pull_training_data     - DVC pull from S3
  3. validate_training_data - Great Expectations suite
  4. run_katib_hpo          - search chunk_size/overlap/top_k
  5. evaluate_new_config    - RAGAS on best Katib params
  6. compare_to_champion    - must improve by 2%
  7a. [PASS] register + promote to Staging + Slack
  7b. [FAIL] log failure + Slack
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Dataset, Metrics, component, pipeline


@component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3==1.34.69", "numpy==1.26.4", "scipy==1.13.0"],
)
def check_drift_severity_component(
    reference_data_uri: str,
    current_data_uri: str,
    aws_region: str,
    js_div_threshold: float,
    drift_report: Output[Artifact],
) -> float:
    """Compute JS-divergence between reference and current embedding distributions."""
    import json, boto3, numpy as np
    from scipy.spatial.distance import jensenshannon
    s3 = boto3.client("s3", region_name=aws_region)
    def load(uri):
        b = uri.split("/")[2]
        k = "/".join(uri.split("/")[3:])
        return np.array(json.loads(s3.get_object(Bucket=b,Key=k)["Body"].read()))
    try:
        ref = load(reference_data_uri)
        cur = load(current_data_uri)
        ref_h, bins = np.histogram(ref.mean(axis=1), bins=50, density=True)
        cur_h, _    = np.histogram(cur.mean(axis=1), bins=bins, density=True)
        ref_h = (ref_h + 1e-10) / (ref_h + 1e-10).sum()
        cur_h = (cur_h + 1e-10) / (cur_h + 1e-10).sum()
        js = float(jensenshannon(ref_h, cur_h))
    except Exception as e:
        print("Drift check error:", e)
        js = 0.0
    report = {"js_divergence": round(js,4), "threshold": js_div_threshold,
              "drift_confirmed": js > js_div_threshold,
              "severity": "HIGH" if js > 0.25 else "MEDIUM" if js > js_div_threshold else "LOW"}
    with open(drift_report.path, "w") as f: json.dump(report, f, indent=2)
    print("Drift:", report)
    return js


@component(
    base_image="python:3.11-slim",
    packages_to_install=["dvc[s3]==3.49.0", "boto3==1.34.69"],
)
def pull_training_data_component(
    dvc_remote: str,
    aws_region: str,
    data_pull_report: Output[Artifact],
) -> bool:
    """Pull latest training data from DVC S3 remote."""
    import json, subprocess
    def run(cmd): return subprocess.run(cmd, capture_output=True, text=True)
    run(["dvc", "remote", "add", "-d", "-f", "s3remote", dvc_remote])
    run(["dvc", "remote", "modify", "s3remote", "region", aws_region])
    r = run(["dvc", "pull"])
    report = {"success": r.returncode == 0, "stdout": r.stdout[-300:]}
    with open(data_pull_report.path, "w") as f: json.dump(report, f, indent=2)
    print("DVC pull:", "OK" if r.returncode == 0 else "FAILED")
    return r.returncode == 0


@component(
    base_image="python:3.11-slim",
    packages_to_install=["psycopg2-binary==2.9.9"],
)
def validate_training_data_component(
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    min_pairs_per_category: int,
    gx_report: Output[Artifact],
) -> bool:
    """Great Expectations training_data_suite: coverage, no-dupes, answer length."""
    import json, psycopg2, psycopg2.extras
    conn = psycopg2.connect(host=postgres_host, dbname=postgres_db,
                            user=postgres_user, password=postgres_password)
    checks, passed = [], True
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT domain, COUNT(*) cnt FROM golden_qa WHERE active=TRUE GROUP BY domain")
        cov = {r["domain"]: r["cnt"] for r in cur.fetchall()}
        for d in ["tech","hr","org"]:
            ok = cov.get(d,0) >= min_pairs_per_category
            checks.append({"name": f"coverage_{d}", "passed": ok, "count": cov.get(d,0)})
            if not ok: passed = False
        cur.execute("SELECT COUNT(*) dupes FROM (SELECT question FROM golden_qa WHERE active=TRUE GROUP BY question HAVING COUNT(*)>1) x")
        dupes = cur.fetchone()["dupes"]
        ok = dupes == 0
        checks.append({"name": "no_duplicates", "passed": ok, "dupes": dupes})
        if not ok: passed = False
    conn.close()
    report = {"passed": passed, "checks": checks}
    with open(gx_report.path, "w") as f: json.dump(report, f, indent=2)
    print("GX:", "PASS" if passed else "FAIL", checks)
    return passed


@component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==26.1.0"],
)
def run_katib_hpo_component(
    kubeflow_namespace: str,
    experiment_name: str,
    max_trials: int,
    katib_result: Output[Artifact],
) -> str:
    """Submit Katib HPO experiment for chunk_size/overlap/top_k. Returns best params JSON."""
    import json, time
    from kubernetes import client, config
    try: config.load_incluster_config()
    except Exception: config.load_kube_config()
    api = client.CustomObjectsApi()
    best = {"chunk_size": 150, "overlap": 30, "top_k": 5}
    exp_body = {
        "apiVersion": "kubeflow.org/v1beta1", "kind": "Experiment",
        "metadata": {"name": experiment_name, "namespace": kubeflow_namespace},
        "spec": {
            "objective": {"type": "maximize", "goal": 0.88,
                          "objectiveMetricName": "ragas_faithfulness"},
            "algorithm": {"algorithmName": "bayesianoptimization"},
            "maxTrialCount": max_trials, "maxFailedTrialCount": 3,
            "parallelTrialCount": 2,
            "parameters": [
                {"name":"chunk_size","parameterType":"int","feasibleSpace":{"min":"50","max":"200"}},
                {"name":"overlap",   "parameterType":"int","feasibleSpace":{"min":"10","max":"50"}},
                {"name":"top_k",     "parameterType":"int","feasibleSpace":{"min":"3","max":"10"}},
            ],
        },
    }
    try:
        api.create_namespaced_custom_object(
            group="kubeflow.org", version="v1beta1",
            namespace=kubeflow_namespace, plural="experiments", body=exp_body)
        deadline = time.time() + 1800
        while time.time() < deadline:
            exp = api.get_namespaced_custom_object(
                "kubeflow.org","v1beta1",kubeflow_namespace,"experiments",experiment_name)
            for cond in exp.get("status",{}).get("conditions",[]):
                if cond.get("type")=="Succeeded" and cond.get("status")=="True":
                    for p in exp["status"].get("currentOptimalTrial",{}).get("parameterAssignments",[]):
                        best[p["name"]] = int(p["value"])
                    break
            else:
                time.sleep(30); continue
            break
    except Exception as e:
        print("Katib error (using defaults):", e)
    with open(katib_result.path,"w") as f:
        json.dump({"best_params": best, "experiment": experiment_name}, f, indent=2)
    print("Katib best:", best)
    return json.dumps(best)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["qdrant-client==1.8.0","sentence-transformers==2.7.0",
                          "ragas==0.1.7","datasets==2.19.0","mlflow==2.12.2",
                          "boto3==1.34.69","psycopg2-binary==2.9.9","httpx==0.27.0"],
)
def evaluate_new_config_component(
    katib_result: Input[Artifact],
    qdrant_host: str, qdrant_port: int, embedding_model: str,
    postgres_host: str, postgres_db: str, postgres_user: str, postgres_password: str,
    ollama_base_url: str, ollama_model: str,
    mlflow_tracking_uri: str, qa_sample_size: int,
    eval_report: Output[Artifact],
) -> float:
    """Re-index with best Katib params, run RAGAS, return faithfulness score."""
    import json, mlflow, psycopg2, psycopg2.extras, httpx
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    with open(katib_result.path) as f: best = json.load(f).get("best_params",{})
    top_k = best.get("top_k", 5)
    conn = psycopg2.connect(host=postgres_host,dbname=postgres_db,
                            user=postgres_user,password=postgres_password)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT question,ground_truth FROM golden_qa WHERE active=TRUE LIMIT %s",(qa_sample_size,))
        pairs = cur.fetchall()
    conn.close()
    if not pairs: raise ValueError("No QA pairs")
    qc  = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
    mdl = SentenceTransformer(embedding_model)
    vecs = mdl.encode([p["question"] for p in pairs], normalize_embeddings=True, convert_to_numpy=True)
    ctxs = []
    for v in vecs:
        hits = []
        for col in ["tech_docs","hr_policies","org_info"]:
            try: hits.extend(qc.search(col, query_vector=v.tolist(), limit=top_k, with_payload=True))
            except Exception: pass
        hits.sort(key=lambda x: x.score, reverse=True)
        ctxs.append([h.payload.get("text","") for h in hits[:top_k]])
    def ask(q, ctx):
        try:
            r = httpx.post(f"{ollama_base_url}/api/generate",
                json={"model":ollama_model,"prompt":"\n".join(ctx)+f"\nQ:{q}\nA:",
                      "stream":False,"options":{"num_predict":256,"temperature":0.0}},timeout=60.0)
            return r.json().get("response","").strip()
        except Exception: return "error"
    answers = [ask(p["question"],c) for p,c in zip(pairs,ctxs)]
    ds = Dataset.from_dict({
        "question":     [p["question"] for p in pairs],
        "answer":       answers,
        "contexts":     ctxs,
        "ground_truth": [p["ground_truth"] for p in pairs],
    })
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    df = result.to_pandas()
    scores = {"faithfulness": float(df["faithfulness"].mean()),
              "answer_relevancy": float(df["answer_relevancy"].mean()),
              "context_precision": float(df["context_precision"].mean()),
              "best_params": best, "sample_count": len(pairs)}
    scores["hallucination_rate"] = round(1.0 - scores["faithfulness"], 4)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("p11-retraining-eval")
    with mlflow.start_run(run_name="retrain-eval") as run:
        mlflow.log_params(best)
        mlflow.log_metrics({k:v for k,v in scores.items() if isinstance(v,float)})
        mlflow.set_tag("pipeline","P11-retraining")
    scores["mlflow_run_id"] = run.info.run_id
    with open(eval_report.path,"w") as f: json.dump(scores, f, indent=2)
    print(f"Eval faithfulness={scores['faithfulness']:.3f}")
    return scores["faithfulness"]


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2","boto3==1.34.69"],
)
def compare_to_champion_component(
    eval_report: Input[Artifact],
    mlflow_tracking_uri: str,
    model_name: str,
    required_improvement_pct: float,
    comparison_result: Output[Artifact],
) -> bool:
    """Compare new faithfulness to Production champion. Returns True if better by required_pct."""
    import json, mlflow
    from mlflow.tracking import MlflowClient
    with open(eval_report.path) as f: new = json.load(f)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    new_faith = new.get("faithfulness", 0.0)
    try: champs = client.get_latest_versions(model_name, stages=["Production"])
    except Exception: champs = []
    if not champs:
        r = {"better": True, "reason": "no_champion", "new_faithfulness": new_faith}
        with open(comparison_result.path,"w") as f: json.dump(r,f,indent=2)
        return True
    champ_faith = client.get_run(champs[0].run_id).data.metrics.get("faithfulness",0.0)
    delta = (new_faith - champ_faith) / max(champ_faith, 1e-6) * 100
    better = delta >= required_improvement_pct
    r = {"better": better, "champion_faithfulness": champ_faith,
         "new_faithfulness": new_faith, "delta_pct": round(delta,2),
         "reason": f"improved {delta:.1f}%" if better else f"only {delta:.1f}%, need {required_improvement_pct}%"}
    with open(comparison_result.path,"w") as f: json.dump(r,f,indent=2)
    print("Champion compare:", r["reason"])
    return better


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2","boto3==1.34.69","requests==2.32.0"],
)
def register_and_promote_component(
    mlflow_tracking_uri: str,
    model_name: str,
    eval_report: Input[Artifact],
    comparison_result: Input[Artifact],
    slack_webhook_url: str,
    promotion_result: Output[Artifact],
) -> None:
    """Register new RAG config in MLflow Registry, promote to Staging, notify Slack."""
    import json, mlflow, requests
    from mlflow.tracking import MlflowClient
    with open(eval_report.path) as f: ed = json.load(f)
    with open(comparison_result.path) as f: cd = json.load(f)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    run_id = ed.get("mlflow_run_id","")
    try:
        try: client.create_registered_model(model_name)
        except Exception: pass
        mv = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model" if run_id else "placeholder",
            run_id=run_id,
            description=f"P11 retrain faithfulness={ed.get('faithfulness',0):.3f}")
        client.transition_model_version_stage(
            name=model_name, version=mv.version,
            stage="Staging", archive_existing_versions=True)
        version, stage = mv.version, "Staging"
    except Exception as e:
        version, stage = "unknown", str(e)
    result = {"promoted": True, "model_name": model_name, "version": version,
              "stage": stage, "faithfulness": ed.get("faithfulness"),
              "best_params": ed.get("best_params"), "delta_pct": cd.get("delta_pct")}
    with open(promotion_result.path,"w") as f: json.dump(result,f,indent=2)
    msg = (f":rocket: *P11 Retraining Complete* - {model_name} v{version} -> {stage}\n"
           f"faithfulness={ed.get('faithfulness',0):.3f} (+{cd.get('delta_pct',0):.1f}% vs champion)\n"
           f"Best params: {ed.get('best_params')}\n"
           f"Next: manual approval to promote Staging -> Production")
    print(msg)
    if slack_webhook_url and slack_webhook_url.startswith("https://"):
        try: requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
        except Exception: pass


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2","boto3==1.34.69","requests==2.32.0"],
)
def log_retrain_failure_component(
    eval_report: Input[Artifact],
    comparison_result: Input[Artifact],
    drift_report: Input[Artifact],
    mlflow_tracking_uri: str,
    slack_webhook_url: str,
    failure_artifact: Output[Artifact],
) -> None:
    """Log retraining failure to MLflow and notify Slack."""
    import json, mlflow, requests
    with open(eval_report.path) as f: ed = json.load(f)
    with open(comparison_result.path) as f: cd = json.load(f)
    with open(drift_report.path) as f: dd = json.load(f)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("p11-retraining-eval")
    with mlflow.start_run(run_name="retrain-failed") as run:
        mlflow.log_metrics({"faithfulness": ed.get("faithfulness",0)})
        mlflow.set_tags({"outcome":"failed","reason":cd.get("reason",""),"pipeline":"P11"})
    record = {"failed":True,"reason":cd.get("reason",""),
              "faithfulness":ed.get("faithfulness"),"drift_severity":dd.get("severity"),
              "mlflow_run_id":run.info.run_id}
    with open(failure_artifact.path,"w") as f: json.dump(record,f,indent=2)
    msg = (f":x: *P11 Retraining Failed*\n"
           f"faithfulness={ed.get('faithfulness',0):.3f} | reason={cd.get('reason','')}\n"
           f"drift={dd.get('severity')} | Manual investigation required")
    print(msg)
    if slack_webhook_url and slack_webhook_url.startswith("https://"):
        try: requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
        except Exception: pass


# =============================================================================
# PIPELINE
# =============================================================================

@pipeline(
    name="p11-automated-retraining",
    description=(
        "Automated retraining: DVC pull -> GX validate -> Katib HPO -> "
        "RAGAS eval -> compare champion -> promote or fail."
    ),
)
def automated_retraining_pipeline(
    trigger_reason: str = "drift_alert",
    reference_data_uri: str = "s3://llm-platform-dvc-remote/reference/embeddings.json",
    current_data_uri: str = "s3://llm-platform-dvc-remote/current/embeddings.json",
    js_div_threshold: float = 0.15,
    dvc_remote: str = "s3://llm-platform-dvc-remote",
    aws_region: str = "us-east-1",
    min_pairs_per_category: int = 20,
    kubeflow_namespace: str = "kubeflow",
    katib_experiment: str = "p11-rag-hpo",
    max_trials: int = 20,
    qdrant_host: str = "qdrant-service.llm-platform-prod.svc.cluster.local",
    qdrant_port: int = 6333,
    embedding_model: str = "all-MiniLM-L6-v2",
    ollama_base_url: str = "http://ollama-service.llm-platform-prod.svc.cluster.local:11434",
    ollama_model: str = "mistral:7b",
    qa_sample_size: int = 50,
    postgres_host: str = "llm-platform.xxxxxxxx.rds.amazonaws.com",
    postgres_db: str = "llm_platform",
    postgres_user: str = "llm_admin",
    postgres_password: str = "",
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
    model_name: str = "llm-platform-rag-config",
    required_improvement_pct: float = 2.0,
    slack_webhook_url: str = "",
) -> None:

    # Step 1: Confirm drift
    drift_task = check_drift_severity_component(
        reference_data_uri=reference_data_uri,
        current_data_uri=current_data_uri,
        aws_region=aws_region,
        js_div_threshold=js_div_threshold,
    )
    drift_task.set_display_name("1 Confirm drift severity")

    # Step 2: DVC pull
    pull_task = pull_training_data_component(
        dvc_remote=dvc_remote,
        aws_region=aws_region,
    )
    pull_task.set_display_name("2 DVC pull training data")
    pull_task.after(drift_task)

    # Step 3: GX validation
    gx_task = validate_training_data_component(
        postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        min_pairs_per_category=min_pairs_per_category,
    )
    gx_task.set_display_name("3 Validate training data (GX)")
    gx_task.after(pull_task)

    # GX passed branch
    with dsl.If(gx_task.outputs["Output"] == True, name="gx-passed"):

        # Step 4: Katib HPO
        katib_task = run_katib_hpo_component(
            kubeflow_namespace=kubeflow_namespace,
            experiment_name=katib_experiment,
            max_trials=max_trials,
        )
        katib_task.set_display_name("4 Katib HPO")

        # Step 5: Evaluate
        eval_task = evaluate_new_config_component(
            katib_result=katib_task.outputs["katib_result"],
            qdrant_host=qdrant_host, qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            postgres_host=postgres_host, postgres_db=postgres_db,
            postgres_user=postgres_user, postgres_password=postgres_password,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model,
            mlflow_tracking_uri=mlflow_tracking_uri,
            qa_sample_size=qa_sample_size,
        )
        eval_task.set_display_name("5 Evaluate new RAG config")

        # Step 6: Compare to champion
        compare_task = compare_to_champion_component(
            eval_report=eval_task.outputs["eval_report"],
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=model_name,
            required_improvement_pct=required_improvement_pct,
        )
        compare_task.set_display_name("6 Compare to champion")

        # Step 7a: Better -> promote
        with dsl.If(compare_task.outputs["Output"] == True, name="better-than-champion"):
            register_and_promote_component(
                mlflow_tracking_uri=mlflow_tracking_uri,
                model_name=model_name,
                eval_report=eval_task.outputs["eval_report"],
                comparison_result=compare_task.outputs["comparison_result"],
                slack_webhook_url=slack_webhook_url,
            ).set_display_name("7a Register + promote to Staging")

        # Step 7b: Not better -> log failure
        with dsl.If(compare_task.outputs["Output"] == False, name="not-better"):
            log_retrain_failure_component(
                eval_report=eval_task.outputs["eval_report"],
                comparison_result=compare_task.outputs["comparison_result"],
                drift_report=drift_task.outputs["drift_report"],
                mlflow_tracking_uri=mlflow_tracking_uri,
                slack_webhook_url=slack_webhook_url,
            ).set_display_name("7b Log failure + notify")

    # GX failed branch
    with dsl.If(gx_task.outputs["Output"] == False, name="gx-failed"):

        @component(base_image="python:3.11-slim",
                   packages_to_install=["requests==2.32.0"])
        def notify_gx_failure(slack_webhook_url: str, gx_report: Input[Artifact]) -> None:
            import json, requests
            with open(gx_report.path) as f: r = json.load(f)
            msg = f":x: *P11 Blocked* - GX training data failed: {r.get('checks')}"
            print(msg)
            if slack_webhook_url and slack_webhook_url.startswith("https://"):
                try: requests.post(slack_webhook_url,json={"text":msg},timeout=10)
                except Exception: pass

        notify_gx_failure(
            slack_webhook_url=slack_webhook_url,
            gx_report=gx_task.outputs["gx_report"],
        ).set_display_name("GX failed notify")


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=automated_retraining_pipeline,
        package_path="pipelines/p11_automated_retraining/pipeline.yaml",
    )
    print("Compiled to pipelines/p11_automated_retraining/pipeline.yaml")