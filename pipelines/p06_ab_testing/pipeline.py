"""
Pipeline 06 - A/B Testing
Compares two workflow variants: ReAct vs Planner-Executor-Critic.
3 queries x 5 iterations = 15 samples per variant.
T-test significance p < 0.05. Quality delta > 5% required.
Logs to MLflow experiment: ab-testing.
Triggered: weekly scheduled (Monday 04:00 UTC).
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Metrics, component, pipeline

AB_QUERIES = '["Why is the payment service timing out?", "What is the leave encashment policy?", "How do I debug a CrashLoopBackOff pod?"]'


@component(
    base_image="python:3.11-slim",
    packages_to_install=["httpx==0.27.0", "mlflow==2.12.2", "boto3==1.34.69"],
)
def run_variant_component(
    variant_name: str,
    workflow_type: str,
    api_base_url: str,
    test_queries_json: str,
    iterations: int,
    mlflow_tracking_uri: str,
    variant_results: Output[Artifact],
) -> float:
    """Run a variant N iterations per query. Returns mean quality score."""
    import json, time, httpx, mlflow
    queries = json.loads(test_queries_json)
    samples = []
    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            try:
                r = httpx.post(
                    f"{api_base_url}/api/query",
                    json={"query": q, "workflow": workflow_type},
                    timeout=90.0)
                latency = round((time.perf_counter() - t0) * 1000, 2)
                ok = r.status_code == 200
                quality = min(1.0, len(r.json().get("response", "")) / 500) if ok else 0.0
                samples.append({"latency_ms": latency, "quality": quality, "success": ok})
            except Exception:
                samples.append({"latency_ms": 90000.0, "quality": 0.0, "success": False})
    avg_latency  = sum(s["latency_ms"] for s in samples) / len(samples)
    avg_quality  = sum(s["quality"]    for s in samples) / len(samples)
    success_rate = sum(1 for s in samples if s["success"]) / len(samples)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("ab-testing")
    with mlflow.start_run(run_name=f"ab-{variant_name}") as run:
        mlflow.log_params({"variant": variant_name, "workflow": workflow_type,
                           "iterations": iterations})
        mlflow.log_metrics({"avg_latency_ms": avg_latency,
                            "avg_quality": avg_quality,
                            "success_rate": success_rate,
                            "sample_count": float(len(samples))})
        mlflow.set_tag("pipeline", "P06-ABTesting")
    result = {
        "variant": variant_name, "workflow_type": workflow_type,
        "samples": samples, "avg_latency_ms": avg_latency,
        "avg_quality": avg_quality, "success_rate": success_rate,
        "mlflow_run_id": run.info.run_id,
    }
    with open(variant_results.path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Variant {variant_name}: latency={avg_latency:.0f}ms quality={avg_quality:.3f}")
    return avg_quality


@component(
    base_image="python:3.11-slim",
    packages_to_install=["scipy==1.13.0", "mlflow==2.12.2",
                          "boto3==1.34.69", "requests==2.32.0"],
)
def declare_winner_component(
    control_results:   Input[Artifact],
    treatment_results: Input[Artifact],
    p_value_threshold:       float,
    quality_delta_threshold: float,
    mlflow_tracking_uri: str,
    slack_webhook_url:   str,
    winner_report: Output[Artifact],
) -> str:
    """T-test between variants. Declare winner if p < threshold AND delta > threshold."""
    import json, mlflow, requests
    from scipy import stats
    with open(control_results.path)   as f: ctrl = json.load(f)
    with open(treatment_results.path) as f: trt  = json.load(f)
    ctrl_q = [s["quality"]    for s in ctrl["samples"]]
    trt_q  = [s["quality"]    for s in trt["samples"]]
    ctrl_l = [s["latency_ms"] for s in ctrl["samples"]]
    trt_l  = [s["latency_ms"] for s in trt["samples"]]
    _, p_q = stats.ttest_ind(ctrl_q, trt_q)
    _, p_l = stats.ttest_ind(ctrl_l, trt_l)
    ctrl_mean = sum(ctrl_q) / len(ctrl_q)
    trt_mean  = sum(trt_q)  / len(trt_q)
    delta = (trt_mean - ctrl_mean) / max(ctrl_mean, 1e-6)
    significant = p_q < p_value_threshold
    meaningful  = abs(delta) > quality_delta_threshold
    if significant and meaningful and delta > 0:
        winner = trt["variant"]
        reason = f"Treatment better by {delta:.1%} (p={p_q:.4f})"
    elif significant and meaningful and delta < 0:
        winner = ctrl["variant"]
        reason = f"Control better by {abs(delta):.1%} (p={p_q:.4f})"
    else:
        winner = "no_winner"
        reason = f"Not significant: p={p_q:.4f} or delta={delta:.1%} too small"
    report = {
        "winner":           winner,
        "reason":           reason,
        "p_value_quality":  round(p_q, 4),
        "p_value_latency":  round(p_l, 4),
        "quality_delta":    round(delta, 4),
        "control":          ctrl["variant"],
        "treatment":        trt["variant"],
    }
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("ab-testing")
    with mlflow.start_run(run_name="ab-winner"):
        mlflow.log_metrics({"p_value_quality": p_q, "quality_delta": delta})
        mlflow.set_tags({"winner": winner, "pipeline": "P06-ABTesting"})
    with open(winner_report.path, "w") as f:
        json.dump(report, f, indent=2)
    msg = f":bar_chart: *P06 A/B Result* Winner: *{winner}*\n{reason}"
    print(msg)
    if slack_webhook_url and slack_webhook_url.startswith("https://"):
        try: requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
        except Exception: pass
    return winner


@pipeline(
    name="p06-ab-testing",
    description="A/B test ReAct vs PEC. T-test significance p<0.05. Weekly scheduled.",
)
def ab_testing_pipeline(
    api_base_url: str = "http://llm-gateway.llm-platform-prod.svc.cluster.local:8000",
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
    control_workflow: str = "react",
    treatment_workflow: str = "planner_executor_critic",
    iterations: int = 5,
    p_value_threshold: float = 0.05,
    quality_delta_threshold: float = 0.05,
    test_queries_json: str = AB_QUERIES,
    slack_webhook_url: str = "",
) -> None:
    control_task = run_variant_component(
        variant_name="control",
        workflow_type=control_workflow,
        api_base_url=api_base_url,
        test_queries_json=test_queries_json,
        iterations=iterations,
        mlflow_tracking_uri=mlflow_tracking_uri)
    control_task.set_display_name(f"1 Run control ({control_workflow})")

    treatment_task = run_variant_component(
        variant_name="treatment",
        workflow_type=treatment_workflow,
        api_base_url=api_base_url,
        test_queries_json=test_queries_json,
        iterations=iterations,
        mlflow_tracking_uri=mlflow_tracking_uri)
    treatment_task.set_display_name(f"2 Run treatment ({treatment_workflow})")

    winner_task = declare_winner_component(
        control_results=control_task.outputs["variant_results"],
        treatment_results=treatment_task.outputs["variant_results"],
        p_value_threshold=p_value_threshold,
        quality_delta_threshold=quality_delta_threshold,
        mlflow_tracking_uri=mlflow_tracking_uri,
        slack_webhook_url=slack_webhook_url)
    winner_task.set_display_name("3 Declare winner (T-test)")


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=ab_testing_pipeline,
        package_path="pipelines/p06_ab_testing/pipeline.yaml")
    print("Compiled P06")