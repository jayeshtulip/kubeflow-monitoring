"""
L1 Infrastructure tests — EKS cluster, H100 node, RDS, Redis, S3.
Run with: pytest tests/l1_infrastructure/ -m l1 -v
"""
from __future__ import annotations
import os
import pytest


@pytest.mark.l1
def test_eks_worker_nodes_ready():
    """Assert 10x c5.4xlarge worker nodes are in Ready state."""
    import boto3
    eks = boto3.client("eks", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    ec2 = boto3.client("ec2", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    cluster_name = os.environ.get("EKS_CLUSTER_NAME", "llm-platform")
    ng = eks.list_nodegroups(clusterName=cluster_name)
    assert len(ng.get("nodegroups", [])) > 0, "No nodegroups found"
    # Check for c5.4xlarge instances
    filters = [
        {"Name": "tag:eks:cluster-name", "Values": [cluster_name]},
        {"Name": "instance-state-name",  "Values": ["running"]},
    ]
    pages = ec2.get_paginator("describe_instances").paginate(Filters=filters)
    instances = []
    for page in pages:
        for res in page["Reservations"]:
            instances.extend(res["Instances"])
    c5_count = sum(1 for i in instances if i.get("InstanceType") == "c5.4xlarge")
    assert c5_count >= 10, f"Expected >= 10 c5.4xlarge nodes, found {c5_count}"


@pytest.mark.l1
def test_h100_node_gpu_allocatable():
    """Assert H100 EC2 instance is running and GPU-enabled."""
    import boto3
    ec2 = boto3.client("ec2", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    filters = [
        {"Name": "instance-state-name", "Values": ["running"]},
        {"Name": "tag:role",            "Values": ["vllm-h100"]},
    ]
    pages = ec2.get_paginator("describe_instances").paginate(Filters=filters)
    h100_instances = []
    for page in pages:
        for res in page["Reservations"]:
            h100_instances.extend(res["Instances"])
    assert len(h100_instances) >= 1, "H100 vLLM instance not found or not running"


@pytest.mark.l1
@pytest.mark.timeout(10)
def test_rds_postgres_connection_latency(env_config, postgres_conn):
    """Assert PostgreSQL connection responds within 50ms."""
    import time
    t0 = time.perf_counter()
    with postgres_conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
    latency_ms = (time.perf_counter() - t0) * 1000
    assert latency_ms < 50.0, f"PostgreSQL latency {latency_ms:.1f}ms > 50ms"


@pytest.mark.l1
@pytest.mark.timeout(10)
def test_redis_cluster_responds(redis_client):
    """Assert Redis responds to PING."""
    result = redis_client.ping()
    assert result is True, "Redis PING failed"


@pytest.mark.l1
def test_s3_dvc_bucket_accessible():
    """Assert DVC S3 remote bucket is accessible with correct IAM policy."""
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    bucket = os.environ.get("DVC_REMOTE_BUCKET", "llm-platform-dvc-remote")
    bucket = bucket.replace("s3://", "").split("/")[0]
    response = s3.head_bucket(Bucket=bucket)
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


@pytest.mark.l1
def test_mlflow_artifacts_bucket_accessible():
    """Assert MLflow S3 artifact bucket is accessible."""
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    bucket = os.environ.get("MLFLOW_S3_ARTIFACT_ROOT",
                            "s3://llm-platform-mlflow-artifacts")
    bucket = bucket.replace("s3://", "").split("/")[0]
    response = s3.head_bucket(Bucket=bucket)
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200