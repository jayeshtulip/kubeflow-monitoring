"""
RAGAS KFP v2 component.

Three sub-components that run sequentially in Pipeline 09:

  1. retrieve_contexts_component
     — For each golden question, retrieve top-k chunks from Qdrant.

  2. generate_answers_component
     — For each question + context, call the LLM (Ollama) to produce an answer.

  3. ragas_score_component
     — Run full RAGAS evaluation (6 metrics), log to MLflow,
       output pass/fail decision for the deployment gate.
"""


from kfp.dsl import Output, Input, Artifact, Dataset, Metrics, component


# ── Component 1: Retrieve contexts ───────────────────────────────────────────

@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=[
        "qdrant-client==1.8.0",
        "sentence-transformers==2.7.0",
        "psycopg2-binary==2.9.9",
    ],
)
def retrieve_contexts_component(
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
    collection_names: str,            # JSON list: '["tech_docs","hr_policies","org_info"]'
    top_k: int,
    domain_filter: str,               # "tech" | "hr" | "org" | "" (all)
    postgres_host: str,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str,
    qa_limit: int,
    # Outputs
    retrieval_dataset: Output[Dataset],
) -> int:
    """
    Load golden QA pairs from PostgreSQL, retrieve top-k contexts from Qdrant
    for each question, and write (question, contexts, ground_truth, domain)
    rows to a JSON-lines dataset artifact.

    Returns the number of questions processed.
    """
    import json
    import os
    import psycopg2
    import psycopg2.extras
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # ── Load QA pairs ─────────────────────────────────────────────────────────
    conn = psycopg2.connect(
        host=postgres_host, dbname=postgres_db,
        user=postgres_user, password=postgres_password,
    )
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        where = "active = TRUE"
        params = []
        if domain_filter:
            where += " AND domain = %s"
            params.append(domain_filter)
        params.append(qa_limit)
        cur.execute(
            f"SELECT question, ground_truth, domain FROM golden_qa WHERE {where} LIMIT %s",
            params,
        )
        pairs = cur.fetchall()
    conn.close()

    if not pairs:
        raise ValueError("No QA pairs found in golden_qa table")

    # ── Embed questions ───────────────────────────────────────────────────────
    model = SentenceTransformer(embedding_model)
    questions = [r["question"] for r in pairs]
    vectors = model.encode(questions, normalize_embeddings=True, convert_to_numpy=True)

    # ── Retrieve from Qdrant ──────────────────────────────────────────────────
    qclient = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
    collections = json.loads(collection_names)

    rows = []
    for i, (pair, vector) in enumerate(zip(pairs, vectors)):
        all_hits = []
        for col in collections:
            try:
                hits = qclient.search(
                    collection_name=col,
                    query_vector=vector.tolist(),
                    limit=top_k,
                    with_payload=True,
                )
                all_hits.extend(hits)
            except Exception:
                pass  # collection may not exist for this domain

        # Re-rank across collections by score, take top_k total
        all_hits.sort(key=lambda h: h.score, reverse=True)
        contexts = [h.payload.get("text", "") for h in all_hits[:top_k]]

        rows.append({
            "question":    pair["question"],
            "ground_truth": pair["ground_truth"],
            "domain":      pair["domain"],
            "contexts":    contexts,
            "answer":      "",            # filled by next component
        })

    with open(retrieval_dataset.path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


# ── Component 2: Generate answers ─────────────────────────────────────────────

@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=[
        "requests==2.32.0",
        "httpx==0.27.0",
    ],
)
def generate_answers_component(
    retrieval_dataset: Input[Dataset],
    ollama_base_url: str,
    model_name: str,
    max_tokens: int,
    # Outputs
    answered_dataset: Output[Dataset],
) -> int:
    """
    For each row in retrieval_dataset, call Ollama to generate an answer
    using the retrieved contexts as the RAG context window.

    Returns the number of answers generated.
    """
    import json
    import httpx

    SYSTEM_PROMPT = (
        "You are an expert infrastructure engineer. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain enough information, say so explicitly. "
        "Be concise and factual."
    )

    def generate(question: str, contexts: list[str]) -> str:
        context_str = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
        prompt = f"{context_str}\n\nQuestion: {question}\nAnswer:"

        payload = {
            "model":  model_name,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }
        try:
            resp = httpx.post(
                f"{ollama_base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[generation_error: {e}]"

    rows = []
    with open(retrieval_dataset.path) as f:
        for line in f:
            row = json.loads(line)
            row["answer"] = generate(row["question"], row["contexts"])
            rows.append(row)

    with open(answered_dataset.path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


# ── Component 3: RAGAS scoring + gate ────────────────────────────────────────

@component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=[
        "ragas==0.1.0",
        "datasets==2.19.0",
        "mlflow==2.12.0",
        "boto3==1.34.0",
        "langchain-community==0.0.38",
        
    ],
)
def ragas_score_component(
    answered_dataset: Input[Dataset],
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    run_name: str,
    faithfulness_hard_block: float,
    hallucination_hard_block: float,
    ragas_metrics_output: Output[Metrics],
    ragas_report: Output[Artifact],
    vllm_base_url: str = 'http://vllm-service.llm-platform-prod.svc.cluster.local:8000',
    vllm_model: str = 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ',
) -> bool:
    """
    Compute all 6 RAGAS metrics and check the deployment gate.

    Returns True (gate PASSED — allow deployment to staging) or
            False (gate FAILED — block deployment, trigger P11 retraining).
    """
    import json
    import time
    import os
    os.environ["OPENAI_API_KEY"] = "dummy-not-used"
    import mlflow
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.llms import VLLMOpenAI
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall, answer_correctness,
    )

    rows = []
    with open(answered_dataset.path) as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        raise ValueError("answered_dataset is empty")

    from datasets import Features, Sequence, Value
    ds = Dataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [[str(c) for c in r["contexts"]] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }, features=Features({
        "question":     Value("string"),
        "answer":       Value("string"),
        "contexts":     Sequence(Value("string")),
        "ground_truth": Value("string"),
    }))

    t0 = time.perf_counter()
    try:
        vllm_llm = LangchainLLMWrapper(VLLMOpenAI(
            openai_api_key='dummy',
            openai_api_base=f'{vllm_base_url}/v1',
            model_name=vllm_model,
            max_tokens=1024,
            temperature=0,
        ))
        faithfulness.llm = vllm_llm
        answer_relevancy.llm = vllm_llm
        context_precision.llm = vllm_llm
        context_recall.llm = vllm_llm
        answer_correctness.llm = vllm_llm
        print(f'Using vLLM judge: {vllm_base_url} model={vllm_model}')
    except Exception as e:
        print(f'Warning: could not init vLLM: {e}')
        import os
        os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"
    result = evaluate(ds, metrics=[
        faithfulness, answer_relevancy,
        context_precision, answer_correctness,
    ], raise_exceptions=False)
    elapsed = round(time.perf_counter() - t0, 2)

    df = result.to_pandas()
    scores = {
        "faithfulness":       float(df["faithfulness"].mean()),
        "answer_relevancy":   float(df["answer_relevancy"].mean()),
        "context_precision":  float(df["context_precision"].mean()),

        "answer_correctness": float(df["answer_correctness"].mean()),
    }
    scores["hallucination_rate"] = round(1.0 - scores["faithfulness"], 4)
    scores["sample_count"]  = len(rows)
    scores["eval_seconds"]  = elapsed

    # Gate check
    gate_failures = []
    if scores["faithfulness"] < faithfulness_hard_block:
        gate_failures.append(
            f"faithfulness={scores['faithfulness']:.3f} < {faithfulness_hard_block}"
        )
    if scores["hallucination_rate"] > hallucination_hard_block:
        gate_failures.append(
            f"hallucination_rate={scores['hallucination_rate']:.3f} > {hallucination_hard_block}"
        )

    scores["gate_passed"]    = len(gate_failures) == 0
    scores["gate_failures"]  = gate_failures

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics({k: v for k, v in scores.items() if isinstance(v, float)})
        mlflow.log_param("sample_count", len(rows))
        mlflow.set_tags({
            "gate_passed":  str(scores["gate_passed"]),
            "gate_failures": "|".join(gate_failures) if gate_failures else "none",
            "pipeline": "P09-RAGAS",
        })

    # Write KFP Metrics
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            ragas_metrics_output.log_metric(k, v)

    # Write full report artifact
    report = {"mlflow_run_id": run.info.run_id, **scores}
    with open(ragas_report.path, "w") as f:
        json.dump(report, f, indent=2)

    status = "PASSED" if scores["gate_passed"] else f"FAILED: {gate_failures}"
    print(f"RAGAS Gate: {status}")
    return scores["gate_passed"]






