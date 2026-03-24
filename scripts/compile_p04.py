import sys
sys.path.insert(0, r'C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS')
from pipelines.p04_quality_monitoring.pipeline import quality_monitoring_pipeline
from kfp import compiler
compiler.Compiler().compile(quality_monitoring_pipeline, 'p04_pipeline.yaml')
print('Compiled: p04_pipeline.yaml')
