import sys
sys.path.insert(0, r"C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS")
from kfp import compiler
from pipelines.p10_dvc_pipeline.pipeline import dvc_reproducibility_pipeline

compiler.Compiler().compile(
    pipeline_func=dvc_reproducibility_pipeline,
    package_path="p10_pipeline.yaml"
)
print("Compiled: p10_pipeline.yaml")
