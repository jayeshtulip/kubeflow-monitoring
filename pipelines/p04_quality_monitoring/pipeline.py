"""
Pipeline 04 - Quality Monitoring
Runs every 15 min. Monitors platform health against SLA thresholds.
  P95 latency < 30s, success_rate > 95%, hallucination_rate < 2%.
Emits: HEALTHY / WARNING / DEGRADED / CRITICAL status.
Logs to MLflow experiment: quality-monitoring.
Feeds Grafana Dashboard 03 (Platform Performance).
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Metrics, component, pipeline


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2", "boto3==1.34.69", "requests==2.32.0"],
)
def collect_platform_metrics_component(
    prometheus_url: str,
    mlflow_tracking_uri: str,
    lookback_minutes: int,
    platform_metrics: Output[Artifact],
) -> str:
    """Collect P95 latency, success rate, hallucination rate from Prometheus + MLflow."""
    import json, requests, mlflow
    metrics = {}
    prom_queries = {
        "p95_latency_ms":  "histogram_quantile(0.95, rate(llm_request_duration_ms_bucket[5m]))",
        "success_rate":    "rate(llm_requests_total{status=\"success\"}[5m]) / rate(llm_requests_total[5m])",
        "request_rate":    "rate(llm_requests_total[5m])",
        "guardrail_rate":  "rate(guardrail_triggered_total[5m])",
    }
    for name, query in prom_queries.items():
        try:
            r = requests.get(f"{prometheus_url}/api/v1/query",
                             params={"query": query}, timeout=10)
            result = r.json().get("data", {}).get("result", [])
            metrics[name] = float(result[0]["value"][1]) if result else 0.0
        except Exception as e:
            metrics[name] = 0.0
            print(f"Prometheus query failed for {name}: {e}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_names=["ragas-evaluation"],
        filter_string="tags.pipeline = 'P09-RAGAS'",
        max_results=1, order_by=["start_time DESC"])
    metrics["hallucination_rate"] = (
        runs[0].data.metrics.get("hallucination_rate", 0.0) if runs else 0.0)
    metrics["ragas_faithfulness"] = (
        runs[0].data.metrics.get("faithfulness", 0.0) if runs else 0.0)
    with open(platform_metrics.path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Collected metrics:", metrics)
    return json.dumps(metrics)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2", "boto3==1.34.69", "requests==2.32.0"],
)
def evaluate_platform_health_component(
    platform_metrics: Input[Artifact],
    p95_latency_threshold_ms: float,
    success_rate_threshold: float,
    hallucination_threshold: float,
    mlflow_tracking_uri: str,
    pagerduty_webhook_url: str,
    slack_webhook_url: str,
    health_report: Output[Artifact],
    health_metrics: Output[Metrics],
) -> str:
    """Evaluate health status. Alert if DEGRADED or CRITICAL."""
    import json, mlflow, requests
    with open(platform_metrics.path) as f:
        m = json.load(f)
    violations = []
    if m.get("p95_latency_ms", 0) > p95_latency_threshold_ms:
        violations.append(
            f"P95 latency {m['p95_latency_ms']:.0f}ms > {p95_latency_threshold_ms}ms")
    if m.get("success_rate", 1.0) < success_rate_threshold:
        violations.append(
            f"Success rate {m['success_rate']:.2%} < {success_rate_threshold:.2%}")
    if m.get("hallucination_rate", 0) > hallucination_threshold:
        violations.append(
            f"Hallucination {m['hallucination_rate']:.3f} > {hallucination_threshold}")
    n = len(violations)
    status = ("HEALTHY"  if n == 0 else
              "WARNING"  if n == 1 else
              "DEGRADED" if n == 2 else
              "CRITICAL")
    health_score = max(0.0, 1.0 - n * 0.33)
    report = {"status": status, "health_score": health_score,
              "violations": violations, "metrics": m}
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("quality-monitoring")
    with mlflow.start_run(run_name=f"health-{status}") as run:
        mlflow.log_metrics({**m, "health_score": health_score})
        mlflow.set_tags({"status": status, "pipeline": "P04-QualityMonitor"})
    report["mlflow_run_id"] = run.info.run_id
    with open(health_report.path, "w") as f:
        json.dump(report, f, indent=2)
    health_metrics.log_metric("health_score", health_score)
    health_metrics.log_metric("violation_count", float(n))
    if status in ("DEGRADED", "CRITICAL"):
        msg = f":fire: *Platform {status}*\n" + "\n".join(violations)
        if slack_webhook_url and slack_webhook_url.startswith("https://"):
            try: requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
            except Exception: pass
        if status == "CRITICAL" and pagerduty_webhook_url.startswith("https://"):
            pd = {"routing_key": "", "event_action": "trigger",
                  "payload": {"summary": msg, "severity": "critical",
                              "source": "llm-platform"}}
            try: requests.post(pagerduty_webhook_url, json=pd, timeout=10)
            except Exception: pass
    print(f"Platform health: {status} score={health_score:.2f}")
    return status


@pipeline(
    name="p04-quality-monitoring",
    description="Monitor platform health every 15 min. Alert on DEGRADED/CRITICAL.",
)
def quality_monitoring_pipeline(
    prometheus_url: str = "http://prometheus-operated.monitoring.svc.cluster.local:9090",
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
    lookback_minutes: int = 15,
    p95_latency_threshold_ms: float = 30000.0,
    success_rate_threshold: float = 0.95,
    hallucination_threshold: float = 0.02,
    pagerduty_webhook_url: str = "",
    slack_webhook_url: str = "",
) -> None:
    collect_task = collect_platform_metrics_component(
        prometheus_url=prometheus_url,
        mlflow_tracking_uri=mlflow_tracking_uri,
        lookback_minutes=lookback_minutes)
    collect_task.set_display_name("1 Collect platform metrics (Prometheus + MLflow)")

    health_task = evaluate_platform_health_component(
        platform_metrics=collect_task.outputs["platform_metrics"],
        p95_latency_threshold_ms=p95_latency_threshold_ms,
        success_rate_threshold=success_rate_threshold,
        hallucination_threshold=hallucination_threshold,
        mlflow_tracking_uri=mlflow_tracking_uri,
        pagerduty_webhook_url=pagerduty_webhook_url,
        slack_webhook_url=slack_webhook_url)
    health_task.set_display_name("2 Evaluate health + alert PagerDuty/Slack")


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=quality_monitoring_pipeline,
        package_path="pipelines/p04_quality_monitoring/pipeline.yaml")
    print("Compiled P04")