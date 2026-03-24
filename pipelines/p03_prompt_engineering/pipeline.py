"""
Pipeline 03 - Prompt Engineering
Tests 6 prompt templates per agent: Planner, Executor, Critic.
Measures response quality, consistency, latency per template.
Logs to MLflow experiment: prompt-engineering.
Output: optimal prompt template ID per agent.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Metrics, component, pipeline

TEMPLATE_TEXTS = {
    "p1_systematic":     "You are a systematic investigator. Create a numbered plan.",
    "p2_chainofthought": "Think step by step before creating your plan.",
    "p3_structured":     "Create a structured plan: Gather, Analyse, Conclude.",
    "p4_minimal":        "Create a concise investigation plan in 3-5 steps.",
    "p5_verbose":        "Create a detailed plan covering all angles.",
    "p6_roleplay":       "You are a senior SRE. Create an investigation plan.",
    "e1_toolselect":     "Select the most appropriate tool for each step.",
    "e2_stepbystep":     "Execute each step carefully before moving to the next.",
    "e3_parallel":       "Identify steps that can run in parallel.",
    "e4_conservative":   "Use only confirmed data sources. Avoid assumptions.",
    "e5_aggressive":     "Gather as much evidence as possible from all sources.",
    "e6_balanced":       "Balance thoroughness with efficiency in tool usage.",
    "c1_strict":         "Reject the investigation if any step lacks evidence.",
    "c2_lenient":        "Accept the investigation if the root cause is clear.",
    "c3_evidencebased":  "Every claim must be backed by tool output evidence.",
    "c4_causal":         "Verify the causal chain is complete and logical.",
    "c5_completeness":   "Check all symptoms are explained by the root cause.",
    "c6_holistic":       "Consider both technical and operational factors.",
}

AGENT_TEMPLATES = {
    "planner":  ["p1_systematic", "p2_chainofthought", "p3_structured",
                 "p4_minimal", "p5_verbose", "p6_roleplay"],
    "executor": ["e1_toolselect", "e2_stepbystep", "e3_parallel",
                 "e4_conservative", "e5_aggressive", "e6_balanced"],
    "critic":   ["c1_strict", "c2_lenient", "c3_evidencebased",
                 "c4_causal", "c5_completeness", "c6_holistic"],
}

TEST_QUERIES = '["Why is the payment service timing out?", "What are pod eviction causes?", "How do I scale a Kubernetes deployment?"]'


@component(
    base_image="python:3.11-slim",
    packages_to_install=["httpx==0.27.0", "mlflow==2.12.2", "boto3==1.34.69"],
)
def test_prompt_template_component(
    agent_type: str,
    template_id: str,
    template_text: str,
    api_base_url: str,
    test_queries_json: str,
    mlflow_tracking_uri: str,
    template_result: Output[Artifact],
) -> float:
    """Test one prompt template. Returns mean quality score."""
    import json, time, httpx, mlflow
    queries = json.loads(test_queries_json)
    results = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                f"{api_base_url}/api/query",
                json={"query": q, "prompt_template": template_id,
                      "agent_type": agent_type},
                timeout=90.0)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            resp = r.json() if r.status_code == 200 else {}
            quality = min(1.0, len(resp.get("response", "")) / 500)
            results.append({"query": q, "quality": quality, "latency_ms": latency})
        except Exception as e:
            results.append({"query": q, "quality": 0.0, "latency_ms": 0.0,
                            "error": str(e)})
    mean_quality  = sum(r["quality"] for r in results) / len(results)
    mean_latency  = sum(r["latency_ms"] for r in results) / len(results)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("prompt-engineering")
    with mlflow.start_run(run_name=f"prompt-{agent_type}-{template_id}") as run:
        mlflow.log_params({"agent_type": agent_type, "template_id": template_id})
        mlflow.log_metrics({"mean_quality": mean_quality,
                            "mean_latency_ms": mean_latency})
        mlflow.set_tag("pipeline", "P03-PromptEngineering")
    report = {"agent_type": agent_type, "template_id": template_id,
              "mean_quality": mean_quality, "mean_latency_ms": mean_latency,
              "results": results, "mlflow_run_id": run.info.run_id}
    with open(template_result.path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Template {agent_type}/{template_id}: quality={mean_quality:.3f}")
    return mean_quality


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2", "boto3==1.34.69"],
)
def select_best_prompts_component(
    mlflow_tracking_uri: str,
    best_prompts_artifact: Output[Artifact],
) -> str:
    """Query MLflow to find best prompt template per agent type."""
    import json, mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    best = {}
    for agent in ["planner", "executor", "critic"]:
        runs = client.search_runs(
            experiment_names=["prompt-engineering"],
            filter_string=f"params.agent_type = '{agent}'",
            order_by=["metrics.mean_quality DESC"],
            max_results=1)
        if runs:
            best[agent] = {
                "template_id":  runs[0].data.params.get("template_id"),
                "mean_quality": runs[0].data.metrics.get("mean_quality"),
            }
    with open(best_prompts_artifact.path, "w") as f:
        json.dump(best, f, indent=2)
    print("Best prompts:", best)
    return json.dumps(best)


@pipeline(
    name="p03-prompt-engineering",
    description="Test 6 templates per agent (planner/executor/critic). 18 total runs.",
)
def prompt_engineering_pipeline(
    api_base_url: str = "http://llm-gateway.llm-platform-prod.svc.cluster.local:8000",
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
    test_queries_json: str = TEST_QUERIES,
) -> None:
    tasks = []
    for agent_type, tmpl_ids in AGENT_TEMPLATES.items():
        for tid in tmpl_ids:
            t = test_prompt_template_component(
                agent_type=agent_type,
                template_id=tid,
                template_text=TEMPLATE_TEXTS.get(tid, ""),
                api_base_url=api_base_url,
                test_queries_json=test_queries_json,
                mlflow_tracking_uri=mlflow_tracking_uri)
            t.set_display_name(f"Test {agent_type}/{tid}")
            tasks.append(t)
    select_task = select_best_prompts_component(
        mlflow_tracking_uri=mlflow_tracking_uri)
    select_task.set_display_name("Select best prompts per agent")
    for t in tasks:
        select_task.after(t)


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=prompt_engineering_pipeline,
        package_path="pipelines/p03_prompt_engineering/pipeline.yaml")
    print("Compiled P03")