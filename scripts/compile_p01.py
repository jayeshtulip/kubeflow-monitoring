import sys
sys.path.insert(0, r'C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS')
from pipelines.p01_model_evaluation.pipeline import model_evaluation_pipeline
from kfp import compiler
compiler.Compiler().compile(model_evaluation_pipeline, 'p01_pipeline.yaml')
print('Compiled: p01_pipeline.yaml')
