"""CloudWatch Logs and Metrics tool for the Executor agent."""
from __future__ import annotations
import json
import time
from datetime import datetime, timedelta
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)


def query_cloudwatch_logs(
    log_group: str,
    query_string: str,
    hours_back: int = 3,
    limit: int = 20,
    cfg: EnvConfig | None = None,
) -> list[dict]:
    """
    Query CloudWatch Logs Insights.
    Returns list of log events matching query_string.
    """
    cfg = cfg or EnvConfig()
    try:
        import boto3
        client = boto3.client("logs", region_name=cfg.aws_region)
        end_time   = int(time.time())
        start_time = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())
        r = client.start_query(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            queryString=query_string,
            limit=limit,
        )
        query_id = r["queryId"]
        # Poll for results
        for _ in range(30):
            time.sleep(2)
            status = client.get_query_results(queryId=query_id)
            if status["status"] == "Complete":
                results = []
                for row in status.get("results", []):
                    record = {field["field"]: field["value"] for field in row}
                    results.append(record)
                logger.info("cloudwatch_logs: %d events from %s", len(results), log_group)
                return results
        return [{"error": "Query timeout", "log_group": log_group}]
    except Exception as exc:
        logger.error("CloudWatch error: %s", exc)
        return [{"error": str(exc), "log_group": log_group, "tool": "cloudwatch_logs"}]


def query_cloudwatch_metrics(
    namespace: str,
    metric_name: str,
    dimensions: list[dict] | None = None,
    hours_back: int = 3,
    cfg: EnvConfig | None = None,
) -> list[dict]:
    """Query CloudWatch Metrics for a given namespace/metric."""
    cfg = cfg or EnvConfig()
    try:
        import boto3
        client = boto3.client("cloudwatch", region_name=cfg.aws_region)
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        kwargs: dict = {
            "Namespace":  namespace,
            "MetricName": metric_name,
            "StartTime":  start_time,
            "EndTime":    end_time,
            "Period":     300,
            "Statistics": ["Average", "Maximum", "Sum"],
        }
        if dimensions:
            kwargs["Dimensions"] = dimensions
        r = client.get_metric_statistics(**kwargs)
        points = sorted(r.get("Datapoints", []), key=lambda x: x["Timestamp"])
        logger.info("cloudwatch_metrics: %d points for %s/%s",
                    len(points), namespace, metric_name)
        return [
            {
                "timestamp": str(p["Timestamp"]),
                "average":   p.get("Average"),
                "maximum":   p.get("Maximum"),
                "sum":       p.get("Sum"),
                "tool":      "cloudwatch_metrics",
            }
            for p in points
        ]
    except Exception as exc:
        logger.error("CloudWatch metrics error: %s", exc)
        return [{"error": str(exc), "tool": "cloudwatch_metrics"}]