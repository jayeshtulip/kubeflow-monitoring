"""Compile P11 auto-retraining pipeline to YAML."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfp import compiler
from pipelines.p11_auto_retraining.pipeline import p11_auto_retraining_pipeline

output = os.path.join(os.path.dirname(__file__), "p11_pipeline.yaml")
compiler.Compiler().compile(
    pipeline_func=p11_auto_retraining_pipeline,
    package_path=output,
)
print(f"Compiled -> {output}")
