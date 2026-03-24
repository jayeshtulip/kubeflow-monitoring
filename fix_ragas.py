f = r"C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS\pipelines\components\ragas\component.py"
with open(f) as fh:
    c = fh.read()

# Fix 1: add langchain-community to packages
c = c.replace(
    '"boto3==1.34.0",\n    ],',
    '"boto3==1.34.0",\n        "langchain-community==0.0.38",\n    ],'
)

# Fix 2: remove duplicate ollama_llm = None line
c = c.replace(
    'ollama_llm = LangchainLLMWrapper(Ollama(base_url=ollama_base_url, model=ollama_model))\n    ollama_llm = None\n    try:\n        from ragas.llms import LangchainLLMWrapper\n        from langchain_community.llms import Ollama\n        ollama_llm = LangchainLLMWrapper(Ollama(base_url=ollama_base_url, model=ollama_model))\n    except Exception as e:\n        print(f\'Warning: could not init Ollama LLM: {e}\')',
    'try:\n        ollama_llm = LangchainLLMWrapper(Ollama(base_url=ollama_base_url, model=ollama_model))\n    except Exception as e:\n        print(f\'Warning: could not init Ollama LLM: {e}\')\n        ollama_llm = None'
)

with open(f, 'w') as fh:
    fh.write(c)
print("Done")
