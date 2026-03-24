
from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Dataset, Metrics, component, pipeline

from pipelines.components.ragas.component import (
    retrieve_contexts_component,
    generate_answers_component,
    ragas_score_component,
)
from pipelines.components.mlflow_registry.component import (
    promote_to_staging_component,
    notify_slack_component,
)


@component(base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
           packages_to_install=["requests==2.32.0"])
def block_deployment_component(
    model_name: str,
    version: str,
    ragas_report: Input[Artifact],
    kubeflow_webhook_url: str,
    block_artifact: Output[Artifact],
) -> None:
    import json, requests
    with open(ragas_report.path) as f:
        ragas_data = json.load(f)
    record = {"blocked": True, "model_name": model_name, "version": version,
              "gate_failures": ragas_data.get("gate_failures", [])}
    with open(block_artifact.path, "w") as f:
        json.dump(record, f, indent=2)
    print("BLOCKED:", model_name, version)
    if kubeflow_webhook_url and kubeflow_webhook_url.startswith("http"):
        try:
            requests.post(kubeflow_webhook_url, json=record, timeout=15)
        except Exception as e:
            print("Webhook error:", e)


@component(base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
           packages_to_install=["psycopg2-binary==2.9.9"])
def check_qa_coverage_component(
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    min_pairs_per_domain: int,
    coverage_report: Output[Artifact],
) -> bool:
    import json, psycopg2, psycopg2.extras
    conn = psycopg2.connect(host=postgres_host, dbname=postgres_db,
                            user=postgres_user, password=postgres_password)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT domain, COUNT(*) as count FROM golden_qa WHERE active=TRUE GROUP BY domain")
        rows = cur.fetchall()
    conn.close()
    coverage = {r["domain"]: r["count"] for r in rows}
    failures = [d for d in ["tech","hr","org"] if coverage.get(d,0) < min_pairs_per_domain]
    report = {"coverage": coverage, "failures": failures, "passed": len(failures)==0}
    with open(coverage_report.path, "w") as f:
        json.dump(report, f, indent=2)
    print("Coverage:", coverage, "Passed:", report["passed"])
    return report["passed"]


@pipeline(
    name="p09-ragas-evaluation",
    description="RAGAS evaluation: retrieve -> generate -> score -> gate -> promote/block",
)
def ragas_evaluation_pipeline(
    domain_filter: str = "",
    qa_limit: int = 200,
    min_pairs_per_domain: int = 20,
    collection_names: str = '["tech_docs","hr_policies","org_info"]',
    top_k: int = 5,
    qdrant_host: str = "qdrant-service.llm-platform-prod.svc.cluster.local",
    qdrant_port: int = 6333,
    embedding_model: str = "all-MiniLM-L6-v2",
    ollama_base_url: str = "http://ollama-service.llm-platform-prod.svc.cluster.local:11434",
    ollama_model: str = "mistral:7b",
    max_tokens: int = 512,
    postgres_host: str = "llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com",
    postgres_db: str = "llm_platform",
    postgres_user: str = "llm_admin",
    postgres_password: str = "Llmplatform2026",
    mlflow_tracking_uri: str = "http://172.20.232.117:80",
    mlflow_experiment: str = "ragas-evaluation",
    run_name: str = "ragas-weekly",
    model_name: str = "llm-platform-rag-config",
    model_version: str = "latest",
    faithfulness_hard_block: float = 0.85,
    hallucination_hard_block: float = 0.15,
    slack_webhook_url: str = "",
    kubeflow_p11_webhook_url: str = "",
) -> None:

    # Step 1: Retrieve contexts
    retrieve_task = retrieve_contexts_component(
        qdrant_host=qdrant_host, qdrant_port=qdrant_port,
        embedding_model=embedding_model, collection_names=collection_names,
        top_k=top_k, domain_filter=domain_filter,
        postgres_host=postgres_host, postgres_db=postgres_db,
        postgres_user=postgres_user, postgres_password=postgres_password,
        qa_limit=qa_limit,
    )
    retrieve_task

    # Step 2: Generate answers
    generate_task = generate_answers_component(
        retrieval_dataset=retrieve_task.outputs["retrieval_dataset"],
        ollama_base_url=ollama_base_url,
        model_name=ollama_model,
        max_tokens=max_tokens,
    )
    generate_task

    # Step 3: RAGAS scoring
    ragas_task = ragas_score_component(
        answered_dataset=generate_task.outputs["answered_dataset"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        run_name=run_name,
        faithfulness_hard_block=faithfulness_hard_block,
        hallucination_hard_block=hallucination_hard_block,
    )
    ragas_task

    # Step 4: GATE PASSED - promote + notify (all inside same context)
    with dsl.If(ragas_task.outputs["Output"] == True, name="gate-passed"):
        promote_task = promote_to_staging_component(
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=model_name,
            version=model_version,
            ragas_gate_passed=True,
            ragas_report=ragas_task.outputs["ragas_report"],
        )
        promote_task

        notify_slack_component(
            slack_webhook_url=slack_webhook_url,
            model_name=model_name,
            version=model_version,
            promotion_result=promote_task.outputs["promotion_result"],
            ragas_report=ragas_task.outputs["ragas_report"],
        )

    # Step 5: GATE FAILED - block + notify (all inside same context)
    with dsl.If(ragas_task.outputs["Output"] == False, name="gate-failed"):
        block_task = block_deployment_component(
            model_name=model_name,
            version=model_version,
            ragas_report=ragas_task.outputs["ragas_report"],
            kubeflow_webhook_url=kubeflow_p11_webhook_url,
        )
        block_task

        notify_slack_component(
            slack_webhook_url=slack_webhook_url,
            model_name=model_name,
            version=model_version,
            promotion_result=block_task.outputs["block_artifact"],
            ragas_report=ragas_task.outputs["ragas_report"],
        )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=ragas_evaluation_pipeline,
        package_path="pipelines/p09_ragas_evaluation/pipeline.yaml",
    )
    print("Compiled to pipelines/p09_ragas_evaluation/pipeline.yaml")




