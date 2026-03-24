"""
End-to-end lineage tracker.
Given a serving pod annotation or MLflow run_id,
traces back through: pod -> registry -> mlflow run -> dvc sha -> raw source docs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pipelines.components.shared.base import EnvConfig, get_logger, get_mlflow_client

logger = get_logger(__name__)


@dataclass
class LineageNode:
    layer: str
    identifier: str
    details: dict = field(default_factory=dict)
    parent: "LineageNode | None" = None

    def to_dict(self) -> dict:
        return {
            "layer":      self.layer,
            "identifier": self.identifier,
            "details":    self.details,
            "parent":     self.parent.to_dict() if self.parent else None,
        }


def trace_from_mlflow_run(
    run_id: str,
    cfg: EnvConfig | None = None,
) -> LineageNode:
    """
    Trace lineage starting from an MLflow run_id.
    Returns a chain: MLflow run -> DVC SHA -> raw data.
    """
    cfg = cfg or EnvConfig()
    client = get_mlflow_client(cfg)

    # MLflow run node
    try:
        run = client.get_run(run_id)
        mlflow_node = LineageNode(
            layer="mlflow_run",
            identifier=run_id,
            details={
                "experiment": run.info.experiment_id,
                "status":     run.info.status,
                "metrics":    run.data.metrics,
                "params":     run.data.params,
                "tags":       run.data.tags,
            }
        )
    except Exception as exc:
        return LineageNode(
            layer="mlflow_run",
            identifier=run_id,
            details={"error": str(exc)},
        )

    # DVC SHA node (from MLflow params)
    dvc_sha = run.data.params.get("dvc_sha", "")
    if dvc_sha:
        dvc_node = LineageNode(
            layer="dvc_artifact",
            identifier=dvc_sha,
            details={
                "remote":    cfg.dvc_remote_bucket,
                "git_ref":   run.data.params.get("git_commit", ""),
            },
            parent=mlflow_node,
        )
        mlflow_node.parent = dvc_node

    # Registry node (which runs referenced this run)
    try:
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            mv = versions[0]
            registry_node = LineageNode(
                layer="model_registry",
                identifier=f"{mv.name} v{mv.version}",
                details={
                    "stage":   mv.current_stage,
                    "created": str(mv.creation_timestamp),
                },
            )
            registry_node.parent = mlflow_node
            logger.info("Lineage traced: registry=%s -> run=%s -> dvc=%s",
                        registry_node.identifier, run_id, dvc_sha)
            return registry_node
    except Exception:
        pass

    return mlflow_node


def trace_from_model_version(
    model_name: str,
    stage: str = "Production",
    cfg: EnvConfig | None = None,
) -> LineageNode | None:
    """Trace lineage from the current Production champion."""
    cfg = cfg or EnvConfig()
    client = get_mlflow_client(cfg)
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            logger.warning("No %s version found for %s", stage, model_name)
            return None
        return trace_from_mlflow_run(versions[0].run_id, cfg)
    except Exception as exc:
        logger.error("Lineage trace failed: %s", exc)
        return None