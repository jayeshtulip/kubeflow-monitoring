f = r"C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS\pipelines\components\ragas\component.py"
with open(f) as fh:
    c = fh.read()

old = """    try:
        ollama_llm = LangchainLLMWrapper(Ollama(base_url=ollama_base_url, model=ollama_model))
    except Exception as e:
        print(f'Warning: could not init Ollama LLM: {e}')
        ollama_llm = None
    result = evaluate(ds, metrics=[
        faithfulness, answer_relevancy,
        context_precision, context_recall, answer_correctness,
    ])"""

new = """    try:
        ollama_llm = LangchainLLMWrapper(Ollama(base_url=ollama_base_url, model=ollama_model))
        faithfulness.llm = ollama_llm
        answer_relevancy.llm = ollama_llm
        context_precision.llm = ollama_llm
        context_recall.llm = ollama_llm
        answer_correctness.llm = ollama_llm
        print(f"Using Ollama LLM: {ollama_base_url} model={ollama_model}")
    except Exception as e:
        print(f'Warning: could not init Ollama LLM: {e}')
        import os
        os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"
    result = evaluate(ds, metrics=[
        faithfulness, answer_relevancy,
        context_precision, context_recall, answer_correctness,
    ])"""

if old in c:
    c = c.replace(old, new)
    with open(f, 'w') as fh:
        fh.write(c)
    print("Fixed")
else:
    print("Pattern not found")
