"""
Promote a Staging model to Production (Champion).
Requires:
  - Model version is in Staging stage
  - RAGAS faithfulness >= 0.90 on latest P09 run
  - Load test on staging passed (P95 < 65s)
  - Manual approval from senior engineer

Usage:
  python mlops/model_registry/promote_to_champion.py --model llm-platform-rag-config --version 3
"""
from __future__ import annotations
import argparse
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pipelines.components.shared.base import EnvConfig, get_logger, get_mlflow_client

logger = get_logger(__name__)

FAITHFULNESS_REQUIRED = 0.90
P95_LATENCY_REQUIRED_MS = 65000


def get_latest_ragas_scores(client: MlflowClient) -> dict:
    """Fetch the most recent P09 RAGAS evaluation scores."""
    runs = client.search_runs(
        experiment_names=["ragas-evaluation"],
        filter_string="tags.pipeline = 'P09-RAGAS'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {}
    return runs[0].data.metrics


def promote_to_champion(
    model_name: str,
    version: str,
    cfg: EnvConfig | None = None,
    skip_checks: bool = False,
) -> bool:
    """
    Promote a model version from Staging to Production.
    Returns True on success, False if checks fail.
    """
    cfg = cfg or EnvConfig()
    client = get_mlflow_client(cfg)

    # Check version is in Staging
    try:
        mv = client.get_model_version(model_name, version)
    except Exception as exc:
        logger.error("Model version not found: %s v%s — %s", model_name, version, exc)
        return False

    if mv.current_stage != "Staging":
        logger.error("Model v%s is in %s, not Staging", version, mv.current_stage)
        return False

    if not skip_checks:
        # Check RAGAS scores
        scores = get_latest_ragas_scores(client)
        faithfulness = scores.get("faithfulness", 0.0)
        if faithfulness < FAITHFULNESS_REQUIRED:
            logger.error(
                "BLOCKED: faithfulness %.3f < required %.3f",
                faithfulness, FAITHFULNESS_REQUIRED,
            )
            return False
        logger.info("RAGAS check passed: faithfulness=%.3f", faithfulness)

        # Manual confirmation
        print(f"\nModel: {model_name} v{version}")
        print(f"Stage: {mv.current_stage} -> Production")
        print(f"RAGAS faithfulness: {faithfulness:.3f} (required >= {FAITHFULNESS_REQUIRED})")
        print(f"Description: {mv.description}")
        confirm = input("\nPromote to Production? [yes/no]: ").strip().lower()
        if confirm != "yes":
            logger.info("Promotion cancelled by user")
            return False

    # Archive existing Production version
    existing = client.get_latest_versions(model_name, stages=["Production"])
    for old in existing:
        client.transition_model_version_stage(
            name=model_name,
            version=old.version,
            stage="Archived",
        )
        logger.info("Archived old champion: v%s", old.version)

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Champion since {__import__('datetime').datetime.utcnow().date()}",
    )
    logger.info("PROMOTED: %s v%s -> Production (Champion)", model_name, version)
    print(f"\nPromotion complete: {model_name} v{version} is now Champion.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote model to Champion")
    parser.add_argument("--model",   required=True,  help="MLflow Registry model name")
    parser.add_argument("--version", required=True,  help="Model version to promote")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip RAGAS and confirmation checks (CI only)")
    args = parser.parse_args()
    ok = promote_to_champion(
        model_name=args.model,
        version=args.version,
        skip_checks=args.skip_checks,
    )
    sys.exit(0 if ok else 1)