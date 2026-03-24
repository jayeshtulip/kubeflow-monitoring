import sys
sys.path.insert(0, r'C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS')
from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline
from kfp import compiler
compiler.Compiler().compile(ragas_evaluation_pipeline, 'p09_pipeline.yaml')
print('Compiled: p09_pipeline.yaml')
