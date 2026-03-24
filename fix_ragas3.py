f = r"C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS\pipelines\components\ragas\component.py"
with open(f) as fh:
    c = fh.read()

old = """    import json
    import time
    import mlflow
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.llms import Ollama
    from ragas.metrics import ("""

new = """    import json
    import time
    import os
    os.environ["OPENAI_API_KEY"] = "dummy-not-used"
    import mlflow
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.llms import Ollama
    from ragas.metrics import ("""

if old in c:
    c = c.replace(old, new)
    with open(f, 'w') as fh:
        fh.write(c)
    print("Fixed")
else:
    print("Pattern not found")
