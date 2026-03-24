"""
MLflow Model Registry KFP v2 component.

Handles model promotion inside Kubeflow pipelines:
  - register_model_component: register a run's artifact as a new model version
  - promote_to_staging_component: transition a version to Staging (gated by RAGAS)
  - promote_to_champion_component: transition Staging → Production (manual approval)

Used by:
  - Pipeline 09 (RAGAS Evaluation) — promotes to Staging if gate passes
  - Pipeline 11 (Automated Retraining) — registers new version post-retrain
"""


from kfp.dsl import Input, Output, Artifact, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.0", "boto3==1.34.0"],
)
def register_model_component(
    mlflow_tracking_uri: str,
    run_id: str,
    model_name: str,
    artifact_path: str,
    # Outputs
    registered_version_artifact: Output[Artifact],
) -> str:
    """
    Register a model artifact from a completed MLflow run into the Model Registry.

    Args:
        run_id:        MLflow run that produced the model artifact.
        model_name:    Registry model name (e.g. "llm-platform-rag-config").
        artifact_path: Path within the run artifacts (e.g. "model").

    Returns:
        The new version string (e.g. "3").
    """
    import json
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Registered from pipeline run {run_id}",
    )

    result = {
        "model_name":    model_name,
        "version":       mv.version,
        "run_id":        run_id,
        "model_uri":     model_uri,
        "current_stage": mv.current_stage,
    }
    with open(registered_version_artifact.path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Registered {model_name} version {mv.version} from run {run_id}")
    return str(mv.version)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.0", "boto3==1.34.0"],
)
def promote_to_staging_component(
    mlflow_tracking_uri: str,
    model_name: str,
    version: str,
    ragas_gate_passed: bool,
    ragas_report: Input[Artifact],
    # Outputs
    promotion_result: Output[Artifact],
) -> bool:
    """
    Promote a registered model version to the Staging stage IF the RAGAS
    gate passed.  Writes a promotion_result artifact recording the decision.

    Returns True if promoted, False if blocked.
    """
    import json
    import mlflow
    from mlflow.tracking import MlflowClient

    with open(ragas_report.path) as f:
        ragas_data = json.load(f)

    if not ragas_gate_passed:
        result = {
            "promoted":      False,
            "reason":        "RAGAS gate failed",
            "gate_failures": ragas_data.get("gate_failures", []),
        }
        with open(promotion_result.path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"BLOCKED: RAGAS gate failed — {ragas_data.get('gate_failures')}")
        return False

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Archive any existing Staging version before promoting
    existing_staging = client.get_latest_versions(model_name, stages=["Staging"])
    for old_mv in existing_staging:
        client.transition_model_version_stage(
            name=model_name,
            version=old_mv.version,
            stage="Archived",
        )
        print(f"Archived old staging version: {old_mv.version}")

    # Promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Promoted to Staging by P09. "
            f"RAGAS faithfulness={ragas_data.get('faithfulness', 'n/a'):.3f}, "
            f"hallucination_rate={ragas_data.get('hallucination_rate', 'n/a'):.3f}"
        ),
    )

    result = {
        "promoted":            True,
        "model_name":          model_name,
        "version":             version,
        "stage":               "Staging",
        "ragas_faithfulness":  ragas_data.get("faithfulness"),
        "ragas_hallucination": ragas_data.get("hallucination_rate"),
        "mlflow_run_id":       ragas_data.get("mlflow_run_id"),
    }
    with open(promotion_result.path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"PROMOTED: {model_name} v{version} → Staging")
    return True


@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.12.0", "boto3==1.34.0", "requests==2.32.0"],
)
def notify_slack_component(
    slack_webhook_url: str,
    model_name: str,
    version: str,
    promotion_result: Input[Artifact],
    ragas_report: Input[Artifact],
) -> None:
    """
    Post a Slack message summarising the RAGAS result and promotion decision.
    Used at the end of P09 regardless of pass/fail.
    """
    import json
    import requests

    with open(promotion_result.path) as f:
        promo = json.load(f)
    with open(ragas_report.path) as f:
        ragas = json.load(f)

    promoted = promo.get("promoted", False)
    icon = ":white_check_mark:" if promoted else ":x:"
    status = "Promoted to *Staging*" if promoted else f"*Blocked* — {promo.get('reason', 'unknown')}"

    text = (
        f"{icon} *P09 RAGAS Evaluation* — {model_name} v{version}\n"
        f"Status: {status}\n"
        f"Faithfulness: `{ragas.get('faithfulness', 'n/a'):.3f}` (target ≥ 0.90)\n"
        f"Hallucination: `{ragas.get('hallucination_rate', 'n/a'):.3f}` (target ≤ 0.10)\n"
        f"Answer Relevancy: `{ragas.get('answer_relevancy', 'n/a'):.3f}`\n"
        f"Samples: {ragas.get('sample_count', 'n/a')} | "
        f"Eval time: {ragas.get('eval_seconds', 'n/a')}s\n"
        f"MLflow run: `{ragas.get('mlflow_run_id', 'n/a')}`"
    )

    if slack_webhook_url and slack_webhook_url.startswith("https://"):
        try:
            requests.post(slack_webhook_url, json={"text": text}, timeout=10)
        except Exception as e:
            print(f"Slack notification failed: {e}")

    print(text)

