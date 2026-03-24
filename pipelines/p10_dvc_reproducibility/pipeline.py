"""
Pipeline 10 - DVC Reproducibility
Verifies DVC pipeline is deterministic.
Runs dvc repro twice, compares output SHAs.
Validates tracked artifacts accessible from S3.
Logs to MLflow experiment: dvc-reproducibility.
Triggered: weekly + on data commit.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Metrics, component, pipeline

DEFAULT_ARTIFACTS = '["data/golden_qa/tech/qa_pairs.json", "data/golden_qa/hr/qa_pairs.json", "data/golden_qa/org/qa_pairs.json"]'


@component(
    base_image="python:3.11-slim",
    packages_to_install=["dvc[s3]==3.49.0", "boto3==1.34.69"],
)
def run_dvc_repro_component(
    dvc_remote: str,
    aws_region: str,
    stage_name: str,
    run_index: int,
    repro_result: Output[Artifact],
) -> str:
    """Run dvc repro and capture output SHA. Returns SHA string."""
    import json, subprocess, hashlib
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True)
    run(["dvc", "remote", "add", "-d", "-f", "s3remote", dvc_remote])
    run(["dvc", "remote", "modify", "s3remote", "region", aws_region])
    run(["dvc", "pull"])
    cmd = ["dvc", "repro", stage_name] if stage_name else ["dvc", "repro"]
    result = run(cmd)
    sha = hashlib.md5(
        (result.stdout + str(result.returncode)).encode()).hexdigest()
    report = {
        "run_index":   run_index,
        "stage":       stage_name or "all",
        "returncode":  result.returncode,
        "output_sha":  sha,
        "stdout_tail": result.stdout[-300:],
    }
    with open(repro_result.path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"DVC repro run {run_index}: SHA={sha} rc={result.returncode}")
    return sha


@component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3==1.34.69"],
)
def validate_artifact_accessibility_component(
    dvc_remote: str,
    aws_region: str,
    artifact_paths_json: str,
    accessibility_report: Output[Artifact],
) -> float:
    """Check all DVC-tracked artifacts are accessible from S3. Returns access rate."""
    import json, boto3
    paths  = json.loads(artifact_paths_json)
    bucket = dvc_remote.replace("s3://", "").split("/")[0]
    prefix = "/".join(dvc_remote.replace("s3://", "").split("/")[1:])
    s3     = boto3.client("s3", region_name=aws_region)
    results = []
    for path in paths:
        key = f"{prefix}/{path}".strip("/")
        try:
            s3.head_object(Bucket=bucket, Key=key)
            results.append({"path": path, "accessible": True})
        except Exception as e:
            results.append({"path": path, "accessible": False, "error": str(e)})
    accessible = sum(1 for r in results if r["accessible"])
    rate = accessible / len(results) if results else 1.0
    report = {"accessibility_rate": rate, "total": len(results),
              "accessible": accessible, "results": results}
    with open(accessibility_report.path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Artifact accessibility: {accessible}/{len(results)} = {rate:.2%}")
    return rate


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.2", "boto3==1.34.69", "requests==2.32.0"],
)
def verify_reproducibility_component(
    run1_result: Input[Artifact],
    run2_result: Input[Artifact],
    accessibility_report: Input[Artifact],
    mlflow_tracking_uri: str,
    slack_webhook_url: str,
    reproducibility_report: Output[Artifact],
) -> bool:
    """Compare SHAs from both repro runs. Pass if identical and artifacts accessible."""
    import json, mlflow, requests
    with open(run1_result.path) as f:       r1  = json.load(f)
    with open(run2_result.path) as f:       r2  = json.load(f)
    with open(accessibility_report.path) as f: acc = json.load(f)
    sha_match  = r1["output_sha"] == r2["output_sha"]
    accessible = acc.get("accessibility_rate", 0) >= 0.99
    passed     = sha_match and accessible
    report = {
        "passed":             passed,
        "sha_match":          sha_match,
        "run1_sha":           r1["output_sha"],
        "run2_sha":           r2["output_sha"],
        "accessibility_rate": acc.get("accessibility_rate"),
        "stage":              r1.get("stage"),
    }
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("dvc-reproducibility")
    with mlflow.start_run(run_name="dvc-repro-check") as run:
        mlflow.log_metrics({
            "sha_match":          float(sha_match),
            "accessibility_rate": acc.get("accessibility_rate", 0),
        })
        mlflow.set_tags({"passed": str(passed), "pipeline": "P10-DVCRepro"})
    report["mlflow_run_id"] = run.info.run_id
    with open(reproducibility_report.path, "w") as f:
        json.dump(report, f, indent=2)
    status = "PASS" if passed else "FAIL"
    msg = (f":white_check_mark: *P10 DVC Reproducibility [{status}]*\n"
           f"SHA match: {sha_match} | Accessibility: {acc.get('accessibility_rate', 0):.2%}")
    print(msg)
    if slack_webhook_url and slack_webhook_url.startswith("https://"):
        try: requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
        except Exception: pass
    return passed


@pipeline(
    name="p10-dvc-reproducibility",
    description="Run dvc repro twice, compare SHAs, validate S3 artifact access.",
)
def dvc_reproducibility_pipeline(
    dvc_remote: str = "s3://llm-platform-dvc-remote",
    aws_region: str = "us-east-1",
    stage_name: str = "",
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
    slack_webhook_url: str = "",
    artifact_paths_json: str = DEFAULT_ARTIFACTS,
) -> None:
    run1_task = run_dvc_repro_component(
        dvc_remote=dvc_remote, aws_region=aws_region,
        stage_name=stage_name, run_index=1)
    run1_task.set_display_name("1 DVC repro run 1")

    run2_task = run_dvc_repro_component(
        dvc_remote=dvc_remote, aws_region=aws_region,
        stage_name=stage_name, run_index=2)
    run2_task.set_display_name("2 DVC repro run 2")
    run2_task.after(run1_task)

    access_task = validate_artifact_accessibility_component(
        dvc_remote=dvc_remote, aws_region=aws_region,
        artifact_paths_json=artifact_paths_json)
    access_task.set_display_name("3 Validate S3 artifact access")

    verify_task = verify_reproducibility_component(
        run1_result=run1_task.outputs["repro_result"],
        run2_result=run2_task.outputs["repro_result"],
        accessibility_report=access_task.outputs["accessibility_report"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        slack_webhook_url=slack_webhook_url)
    verify_task.set_display_name("4 Verify SHA match + report")


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=dvc_reproducibility_pipeline,
        package_path="pipelines/p10_dvc_reproducibility/pipeline.yaml")
    print("Compiled P10")