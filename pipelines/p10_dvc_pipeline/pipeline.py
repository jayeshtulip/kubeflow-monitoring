from kfp import dsl
from kfp.dsl import Output, Artifact, component, pipeline


@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["dvc[s3]==3.50.0", "boto3==1.34.0"]
)
def dvc_pull_component(
    s3_remote: str,
    aws_region: str,
    pull_report: Output[Artifact],
) -> int:
    import subprocess, json, os
    os.environ["AWS_DEFAULT_REGION"] = aws_region

    import subprocess as sp2, os, pathlib
    sp2.run(["apt-get", "update", "-q"], capture_output=True)
    sp2.run(["apt-get", "install", "-y", "-q", "git"], capture_output=True)
    clone2 = sp2.run(["git", "clone", "--depth", "1",
        "https://github.com/jayeshtulip/kubeflow-monitoring.git",
        "/workspace"], capture_output=True, text=True)
    print("Pull clone:", clone2.returncode, clone2.stderr[-100:])
    os.chdir("/workspace")
    # Configure DVC remote
    subprocess.run(["dvc", "remote", "add", "-d", "s3remote", s3_remote, "--force"],
                   capture_output=True)
    subprocess.run(["dvc", "remote", "modify", "s3remote", "region", aws_region],
                   capture_output=True)

    result = subprocess.run(["dvc", "pull", "--force"],
                            capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    report = {"status": "success" if result.returncode == 0 else "failed",
              "stdout": result.stdout[-500:], "returncode": result.returncode}
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
    repro_report: Output[Artifact],
) -> str:
    import subprocess, json, os
    os.environ["AWS_DEFAULT_REGION"] = aws_region
    os.environ["QDRANT_HOST"] = qdrant_host
    os.environ["POSTGRES_HOST"] = postgres_host
    os.environ["POSTGRES_DB"] = postgres_db
    os.environ["POSTGRES_USER"] = postgres_user
    os.environ["POSTGRES_PASSWORD"] = postgres_password

    import subprocess as sp
    sp.run(["apt-get", "update", "-q"], capture_output=True)
    git_inst = sp.run(["apt-get", "install", "-y", "-q", "git"], capture_output=True, text=True)
    print("git install:", git_inst.returncode, git_inst.stderr[-100:])
    clone = sp.run(["git", "clone", "--depth", "1",
        "https://github.com/jayeshtulip/kubeflow-monitoring.git",
        "/workspace"], capture_output=True, text=True)
    print("Clone:", clone.returncode, clone.stdout, clone.stderr)
    os.chdir("/workspace")
    sp.run(["dvc", "remote", "add", "-d", "s3remote", s3_remote, "--force"], capture_output=True)
    sp.run(["dvc", "remote", "modify", "s3remote", "region", aws_region], capture_output=True)
    pull_result = sp.run(["dvc", "pull", "--force"], capture_output=True, text=True)
    print("DVC pull stdout:", pull_result.stdout[-300:])
    print("DVC pull stderr:", pull_result.stderr[-300:])
    print("DVC pull returncode:", pull_result.returncode)
    # Create data/raw if dvc pull did not restore it
    import pathlib
    if not pathlib.Path("/workspace/data/raw").exists():
        print("data/raw missing after dvc pull - pulling specific target")
        sp.run(["dvc", "pull", "data/raw"], capture_output=True)
        pathlib.Path("/workspace/data/raw").mkdir(parents=True, exist_ok=True)
    print(f"Running: dvc repro {stage}")
    if stage == "all":
        result = subprocess.run(["dvc", "repro"],
                                capture_output=True, text=True)
    else:
        result = subprocess.run(["dvc", "repro", stage],
                                capture_output=True, text=True)

    print(result.stdout[-1000:])
    print(result.stderr[-500:])

    status = "success" if result.returncode == 0 else "failed"
    report = {"stage": stage, "status": status,
              "stdout": result.stdout[-500:], "returncode": result.returncode}
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
    push_report: Output[Artifact],
) -> str:
    import subprocess, json, os, hashlib
    os.environ["AWS_DEFAULT_REGION"] = aws_region

    # Push artifacts to S3
    result = subprocess.run(["dvc", "push"],
                            capture_output=True, text=True)
    print(result.stdout)

    # Get DVC SHA for lineage
    sha_result = subprocess.run(["dvc", "status", "--json"],
                                capture_output=True, text=True)
    dvc_sha = hashlib.md5(sha_result.stdout.encode()).hexdigest()[:8]

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("dvc-pipeline")
        with mlflow.start_run(run_name="dvc-repro"):
            mlflow.log_param("dvc_sha", dvc_sha)
            mlflow.log_param("s3_remote", s3_remote)
            mlflow.set_tag("pipeline", "P10-DVC")
        print(f"Logged to MLflow: dvc_sha={dvc_sha}")
    except Exception as e:
        print(f"MLflow log failed (non-critical): {e}")

    status = "success" if result.returncode == 0 else "failed"
    report = {"status": status, "dvc_sha": dvc_sha, "push_returncode": result.returncode}
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
) -> None:

    # Step 1: Pull latest data from S3
    pull_task = dvc_pull_component(
        s3_remote=s3_remote,
        aws_region=aws_region,
    )

    # Step 2: Run dvc repro inside EKS cluster
    repro_task = dvc_repro_component(
        stage=stage,
        s3_remote=s3_remote,
        aws_region=aws_region,
        qdrant_host=qdrant_host,
        postgres_host=postgres_host,
        postgres_db=postgres_db,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
    )
    repro_task.after(pull_task)

    # Step 3: Push artifacts back to S3 + log SHA to MLflow
    push_task = dvc_push_component(
        s3_remote=s3_remote,
        aws_region=aws_region,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    push_task.after(repro_task)
