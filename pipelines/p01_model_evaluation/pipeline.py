"""
P01: Model Evaluation Pipeline

Tests all 4 LangGraph workflows (Simple, Tools, Smart, ReAct)
with 5 queries each = 20 total test cases.
Logs results to MLflow experiment 'workflow-evaluation'.
Trigger: Daily scheduled / on-demand
"""
from kfp import dsl
from kfp.dsl import component, Output, Input, Artifact, Metrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["httpx==0.27.0", "mlflow==2.12.2"],
)
def run_workflow_tests_component(
    api_base_url: str,
    workflow_type: str,
    test_queries_json: str,
    timeout_seconds: int,
    mlflow_tracking_uri: str,
    test_results: Output[Artifact],
    test_metrics: Output[Metrics],
) -> float:
    """Run test queries against a specific LangGraph workflow type."""
    import json, time, mlflow
    import httpx

    queries = json.loads(test_queries_json)
    results = []
    latencies = []

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("workflow-evaluation")

    with mlflow.start_run(run_name=f"{workflow_type}-evaluation"):
        for query in queries:
            try:
                t0 = time.perf_counter()
                resp = httpx.post(
                    f"{api_base_url}/query",
                    json={"query": query, "workflow_hint": workflow_type},
                    timeout=timeout_seconds,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                success = resp.status_code == 200
                results.append({"query": query, "success": success, "latency_ms": latency_ms})
                latencies.append(latency_ms)
            except Exception as exc:
                results.append({"query": query, "success": False, "latency_ms": timeout_seconds * 1000, "error": str(exc)})
                latencies.append(timeout_seconds * 1000)

        success_rate = sum(r["success"] for r in results) / len(results)
        p95_latency  = sorted(latencies)[int(len(latencies) * 0.95)]
        avg_latency  = sum(latencies) / len(latencies)

        metrics = {
            "success_rate": success_rate,
            "p95_latency_ms": p95_latency,
            "avg_latency_ms": avg_latency,
            "total_queries": len(results),
        }

        mlflow.log_metrics(metrics)
        mlflow.set_tag("workflow_type", workflow_type)

        report = {"workflow_type": workflow_type, "metrics": metrics, "results": results}
        with open(test_results.path, "w") as f:
            json.dump(report, f, indent=2)
        for k, v in metrics.items():
            test_metrics.log_metric(k, v)

        print(f"Workflow {workflow_type}: success={success_rate:.2%} p95={p95_latency:.0f}ms")
        return success_rate


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2"],
)
def compare_workflows_component(
    simple_results:  Input[Artifact],
    tools_results:   Input[Artifact],
    smart_results:   Input[Artifact],
    react_results:   Input[Artifact],
    mlflow_tracking_uri: str,
    comparison_report: Output[Artifact],
) -> str:
    """Aggregate all 4 workflow results. Returns name of best workflow."""
    import json, mlflow

    workflows = {}
    for name, artifact in [
        ("simple", simple_results),
        ("tools",  tools_results),
        ("smart",  smart_results),
        ("react",  react_results),
    ]:
        try:
            with open(artifact.path) as f:
                workflows[name] = json.load(f)
        except Exception as exc:
            workflows[name] = {"metrics": {"success_rate": 0, "p95_latency_ms": 999999, "avg_latency_ms": 999999, "total_queries": 0}, "error": str(exc)}

    best = max(workflows, key=lambda w: workflows[w]["metrics"]["success_rate"])
    comparison = {
        "best_workflow": best,
        "workflows": {k: v["metrics"] for k, v in workflows.items()},
    }

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("workflow-evaluation")
    with mlflow.start_run(run_name="workflow-comparison"):
        for wf, data in workflows.items():
            for k, v in data["metrics"].items():
                mlflow.log_metric(f"{wf}_{k}", v)
        mlflow.set_tag("best_workflow", best)

    with open(comparison_report.path, "w") as fh:
        json.dump(comparison, fh, indent=2)

    print(f"Best workflow: {best}")
    return best


@dsl.pipeline(
    name="p01-model-evaluation",
    description="Test all 4 LangGraph workflows. 5 queries x 4 = 20 test cases.",
)
def model_evaluation_pipeline(
    api_base_url: str = "http://llm-gateway.llm-platform-prod.svc.cluster.local:8000",
    mlflow_tracking_uri: str = "http://172.20.232.117:80",
    test_queries_json: str = '["Why is the payment service timing out?", "What is the leave policy?", "How do I restart a pod?", "What are the RDS connection limits?"]',
    timeout_seconds: int = 90,
):
    simple_task = run_workflow_tests_component(
        api_base_url=api_base_url, workflow_type="simple",
        test_queries_json=test_queries_json, timeout_seconds=timeout_seconds,
        mlflow_tracking_uri=mlflow_tracking_uri)

    tools_task = run_workflow_tests_component(
        api_base_url=api_base_url, workflow_type="tools",
        test_queries_json=test_queries_json, timeout_seconds=timeout_seconds,
        mlflow_tracking_uri=mlflow_tracking_uri)

    smart_task = run_workflow_tests_component(
        api_base_url=api_base_url, workflow_type="smart",
        test_queries_json=test_queries_json, timeout_seconds=timeout_seconds,
        mlflow_tracking_uri=mlflow_tracking_uri)

    react_task = run_workflow_tests_component(
        api_base_url=api_base_url, workflow_type="react",
        test_queries_json=test_queries_json, timeout_seconds=timeout_seconds,
        mlflow_tracking_uri=mlflow_tracking_uri)

    compare_workflows_component(
        simple_results=simple_task.outputs["test_results"],
        tools_results=tools_task.outputs["test_results"],
        smart_results=smart_task.outputs["test_results"],
        react_results=react_task.outputs["test_results"],
        mlflow_tracking_uri=mlflow_tracking_uri)


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=model_evaluation_pipeline,
        package_path="pipelines/p01_model_evaluation/pipeline.yaml")
    print("Compiled P01")
