# Promote MLflow run to Staging in Model Registry
import mlflow

def promote_to_staging(run_id: str, model_name: str):
    client = mlflow.tracking.MlflowClient()
    mv = client.create_model_version(model_name, f'runs:/{run_id}/model', run_id)
    client.transition_model_version_stage(model_name, mv.version, 'Staging')
    print(f'Promoted {model_name} v{mv.version} to Staging')
