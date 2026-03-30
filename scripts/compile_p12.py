import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kfp import compiler
from pipelines.p12_e2e_test.pipeline import p12_e2e_pipeline
output = os.path.join(os.path.dirname(__file__), "p12_pipeline.yaml")
compiler.Compiler().compile(pipeline_func=p12_e2e_pipeline, package_path=output)
print(f"Compiled -> {output}")
