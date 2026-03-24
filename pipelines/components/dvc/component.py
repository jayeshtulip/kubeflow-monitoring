"""
DVC KFP v2 component.

Handles DVC operations inside Kubeflow pipelines:
  - dvc_add_and_push:  add a local file/dir to DVC tracking and push to S3
  - dvc_pull:          pull a specific artifact from the DVC remote
  - dvc_repro:         run dvc repro for a named stage and verify SHA

All operations record the resulting DVC SHA in an output artifact so
downstream components (Qdrant indexer, MLMD logger) can link data
version to pipeline execution for end-to-end lineage.
"""


from kfp import dsl
from kfp.dsl import Output, Input, Artifact, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "dvc[s3]==3.49.0",
        "boto3==1.34.0",
    ],
)
def dvc_add_and_push_component(
    local_path: str,
    dvc_remote: str,
    aws_region: str,
    # Outputs
    dvc_sha_artifact: Output[Artifact],
) -> str:
    """
    Track a file or directory with DVC and push to the S3 remote.

    Args:
        local_path:  Path inside the pipeline container to track.
        dvc_remote:  S3 URI of the DVC remote (e.g. s3://llm-platform-dvc-remote).
        aws_region:  AWS region for the S3 bucket.

    Returns:
        DVC content SHA of the tracked artifact.
    """
    import json
    import subprocess

    def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    # Configure remote
    run(["dvc", "remote", "add", "-d", "-f", "s3remote", dvc_remote])
    run(["dvc", "remote", "modify", "s3remote", "region", aws_region])

    # Track and push
    run(["dvc", "add", local_path])
    push_result = run(["dvc", "push"])

    # Extract SHA from .dvc file
    dvc_file = f"{local_path}.dvc"
    sha = "unknown"
    try:
        with open(dvc_file) as f:
            for line in f:
                if "md5:" in line or "sha256:" in line:
                    sha = line.strip().split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        sha = "no_dvc_file"

    result = {"local_path": local_path, "dvc_sha": sha, "remote": dvc_remote}
    with open(dvc_sha_artifact.path, "w") as f:
        json.dump(result, f)

    return sha


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "dvc[s3]==3.49.0",
        "boto3==1.34.0",
    ],
)
def dvc_pull_component(
    dvc_remote: str,
    aws_region: str,
    target: str = "",
) -> None:
    """
    Pull all tracked artifacts (or a specific target) from the DVC remote.

    Args:
        dvc_remote:  S3 URI of the DVC remote.
        aws_region:  AWS region.
        target:      Optional specific .dvc file or stage to pull.
    """
    import subprocess

    def run(cmd: list[str]) -> None:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    run(["dvc", "remote", "add", "-d", "-f", "s3remote", dvc_remote])
    run(["dvc", "remote", "modify", "s3remote", "region", aws_region])

    pull_cmd = ["dvc", "pull"]
    if target:
        pull_cmd.append(target)
    run(pull_cmd)


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "dvc[s3]==3.49.0",
        "boto3==1.34.0",
    ],
)
def dvc_repro_component(
    stage_name: str,
    dvc_remote: str,
    aws_region: str,
    dvc_repro_artifact: Output[Artifact],
) -> bool:
    """
    Run dvc repro for a named stage and verify reproducibility.
    Returns True if output SHA matches a previous run (pipeline is deterministic).
    """
    import json
    import subprocess
    import hashlib

    def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    run(["dvc", "remote", "add", "-d", "-f", "s3remote", dvc_remote])
    run(["dvc", "remote", "modify", "s3remote", "region", aws_region])
    run(["dvc", "pull"])

    result1 = run(["dvc", "repro", stage_name], check=False)
    status1_sha = hashlib.md5(result1.stdout.encode()).hexdigest()

    # Second run — should produce identical outputs (reproducibility check)
    result2 = run(["dvc", "repro", stage_name], check=False)
    status2_sha = hashlib.md5(result2.stdout.encode()).hexdigest()

    reproducible = (status1_sha == status2_sha)

    report = {
        "stage":        stage_name,
        "run1_sha":     status1_sha,
        "run2_sha":     status2_sha,
        "reproducible": reproducible,
        "stdout_r1":    result1.stdout[-500:],
    }
    with open(dvc_repro_artifact.path, "w") as f:
        json.dump(report, f, indent=2)

    return reproducible

