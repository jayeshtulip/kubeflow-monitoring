from kfp import dsl
from kfp.dsl import Output, Artifact, component, pipeline


@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["dvc[s3]==3.50.0", "boto3==1.34.0"]
)
def dvc_pull_component(
    s3_remote: str,
    aws_region: str,
    run_uuid: str,
    pull_report: Output[Artifact],
) -> int:
    import subprocess, json, os
    os.environ["AWS_DEFAULT_REGION"] = aws_region
    print(f"Run UUID: {run_uuid}")

    import subprocess as sp
    sp.run(["apt-get", "update", "-q"], capture_output=True)
    sp.run(["apt-get", "install", "-y", "-q", "git"], capture_output=True)
    clone = sp.run(["git", "clone", "--depth", "1",
        "https://github.com/jayeshtulip/kubeflow-monitoring.git",
        "/workspace"], capture_output=True, text=True)
    print("Clone:", clone.returncode, clone.stderr[-100:])
    os.chdir("/workspace")
    subprocess.run(["dvc", "remote", "add", "-d", "s3remote", s3_remote, "--force"], capture_output=True)
    subprocess.run(["dvc", "remote", "modify", "s3remote", "region", "us-east-1"], capture_output=True)
    result = subprocess.run(["dvc", "pull", "--force"], capture_output=True, text=True)
    print("DVC pull stdout:", result.stdout[-300:])
    print("DVC pull stderr:", result.stderr[-300:])
    print("DVC pull returncode:", result.returncode)

    report = {"status": "success" if result.returncode == 0 else "failed",
              "stdout": result.stdout[-500:], "returncode": result.returncode,
              "run_uuid": run_uuid}
    with open(pull_report.path, "w") as f:
        json.dump(report, f)
    return result.returncode


@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["dvc[s3]==3.50.0", "pathspec==0.11.2", "boto3==1.34.0"]
)
def dvc_repro_component(
    stage: str,
    s3_remote: str,
    aws_region: str,
    qdrant_host: str,
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    run_uuid: str,
    repro_report: Output[Artifact],
) -> str:
    import subprocess, json, os, pathlib
    os.environ["AWS_DEFAULT_REGION"] = aws_region
    os.environ["QDRANT_HOST"] = qdrant_host
    os.environ["POSTGRES_HOST"] = postgres_host
    os.environ["POSTGRES_DB"] = postgres_db
    os.environ["POSTGRES_USER"] = postgres_user
    os.environ["POSTGRES_PASSWORD"] = postgres_password
    print(f"Run UUID: {run_uuid}")

    import subprocess as sp
    sp.run(["apt-get", "update", "-q"], capture_output=True)
    sp.run(["apt-get", "install", "-y", "-q", "git"], capture_output=True)
    clone = sp.run(["git", "clone", "--depth", "1",
        "https://github.com/jayeshtulip/kubeflow-monitoring.git",
        "/workspace"], capture_output=True, text=True)
    print("Clone:", clone.returncode, clone.stderr[-50:])
    os.chdir("/workspace")
    sp.run(["dvc", "remote", "add", "-d", "s3remote", s3_remote, "--force"], capture_output=True)
    sp.run(["dvc", "remote", "modify", "s3remote", "region", "us-east-1"], capture_output=True)
    pull = sp.run(["dvc", "pull", "--force"], capture_output=True, text=True)
    print("DVC pull:", pull.returncode, pull.stdout[-200:])

    if not pathlib.Path("/workspace/data/raw").exists():
        pathlib.Path("/workspace/data/raw").mkdir(parents=True, exist_ok=True)

    print(f"Running: dvc repro {stage}")
    if stage == "all":
        result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
    else:
        result = subprocess.run(["dvc", "repro", stage], capture_output=True, text=True)

    print(result.stdout[-1000:])
    print(result.stderr[-500:])

    status = "success" if result.returncode == 0 else "failed"
    report = {"stage": stage, "status": status,
              "stdout": result.stdout[-500:], "returncode": result.returncode,
              "run_uuid": run_uuid}
    with open(repro_report.path, "w") as f:
        json.dump(report, f)
    return status


@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["dvc[s3]==3.50.0", "boto3==1.34.0"]
)
def dvc_push_component(
    s3_remote: str,
    aws_region: str,
    mlflow_tracking_uri: str,
    run_uuid: str,
    push_report: Output[Artifact],
) -> str:
    import subprocess, json, os, hashlib
    os.environ["AWS_DEFAULT_REGION"] = aws_region
    print(f"Run UUID: {run_uuid}")

    result = subprocess.run(["dvc", "push"], capture_output=True, text=True)
    print(result.stdout)

    sha_result = subprocess.run(["dvc", "status", "--json"], capture_output=True, text=True)
    dvc_sha = hashlib.md5(sha_result.stdout.encode()).hexdigest()[:8]

    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("dvc-pipeline")
        with mlflow.start_run(run_name=f"dvc-repro-{run_uuid[:8]}"):
            mlflow.log_param("dvc_sha", dvc_sha)
            mlflow.log_param("s3_remote", s3_remote)
            mlflow.log_param("run_uuid", run_uuid)
            mlflow.set_tag("pipeline", "P10-DVC")
        print(f"Logged to MLflow: dvc_sha={dvc_sha}")
    except Exception as e:
        print(f"MLflow log failed (non-critical): {e}")

    status = "success" if result.returncode == 0 else "failed"
    report = {"status": status, "dvc_sha": dvc_sha,
              "push_returncode": result.returncode, "run_uuid": run_uuid}
    with open(push_report.path, "w") as f:
        json.dump(report, f)
    return status


@pipeline(
    name="p10-dvc-reproducibility",
    description="DVC repro inside EKS - preprocess, embed, index, validate, push to S3"
)
def dvc_reproducibility_pipeline(
    s3_remote: str = "s3://llm-platform-dvc-remote-659071697671/dvc",
    aws_region: str = "us-east-1",
    qdrant_host: str = "qdrant.llm-platform-prod.svc.cluster.local",
    postgres_host: str = "llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com",
    postgres_db: str = "llm_platform",
    postgres_user: str = "llm_admin",
    postgres_password: str = "Llmplatform2026",
    mlflow_tracking_uri: str = "http://172.20.172.203:5000",
    stage: str = "all",
    run_uuid: str = "default",  # pass timestamp from submit script to bust cache
) -> None:

    pull_task = dvc_pull_component(
        s3_remote=s3_remote,
        aws_region=aws_region,
        run_uuid=run_uuid,
    )
    pull_task.set_caching_options(False)

    repro_task = dvc_repro_component(
        stage=stage,
        s3_remote=s3_remote,
        aws_region=aws_region,
        qdrant_host=qdrant_host,
        postgres_host=postgres_host,
        postgres_db=postgres_db,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        run_uuid=run_uuid,
    )
    repro_task.set_caching_options(False)
    repro_task.after(pull_task)

    push_task = dvc_push_component(
        s3_remote=s3_remote,
        aws_region=aws_region,
        mlflow_tracking_uri=mlflow_tracking_uri,
        run_uuid=run_uuid,
    )
    push_task.set_caching_options(False)
    push_task.after(repro_task)
