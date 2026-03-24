"""
RAGAS evaluator.

Computes all 6 production metrics against a golden QA dataset:
  1. faithfulness        — claims grounded in retrieved context (NLI)
  2. answer_relevancy    — response semantically answers the question
  3. context_precision   — fraction of retrieved chunks that are relevant
  4. context_recall      — fraction of ground-truth facts covered by context
  5. answer_correctness  — factual + semantic accuracy vs ground truth
  6. hallucination_rate  — derived: 1 - faithfulness (convenience metric)

Production thresholds (regression gate in P09):
  faithfulness       >= 0.90   (hard block if < 0.85)
  answer_relevancy   >= 0.85
  context_precision  >= 0.80
  context_recall     >= 0.75
  answer_correctness >= 0.80
  hallucination_rate <= 0.10  (hard block if > 0.15)

Used by:
  - Kubeflow Pipeline 09 (RAGAS Evaluation) — primary consumer
  - Kubeflow Pipeline 11 (Automated Retraining) — post-train evaluation
  - Grafana Dashboard 02 (MLflow Metrics) — reads scores from MLflow
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from pipelines.components.shared.base import (
    EnvConfig,
    get_logger,
    get_mlflow_client,
)

logger = get_logger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness":       {"target": 0.90, "hard_block": 0.85},
    "answer_relevancy":   {"target": 0.85, "hard_block": 0.75},
    "context_precision":  {"target": 0.80, "hard_block": 0.70},
    "context_recall":     {"target": 0.75, "hard_block": 0.65},
    "answer_correctness": {"target": 0.80, "hard_block": 0.70},
    "hallucination_rate": {"target": 0.10, "hard_block": 0.15, "lower_is_better": True},
}

ALL_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RAGASInput:
    """Single evaluation sample."""
    question: str
    answer: str                    # LLM-generated answer
    contexts: list[str]            # Retrieved chunks
    ground_truth: str              # Golden answer from dataset
    domain: str = "general"        # tech / hr / org


@dataclass
class RAGASScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float
    hallucination_rate: float      # = 1 - faithfulness
    sample_count: int
    eval_seconds: float
    passed_gate: bool
    gate_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness":       round(self.faithfulness, 4),
            "answer_relevancy":   round(self.answer_relevancy, 4),
            "context_precision":  round(self.context_precision, 4),
            "context_recall":     round(self.context_recall, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "sample_count":       self.sample_count,
            "eval_seconds":       round(self.eval_seconds, 2),
            "passed_gate":        self.passed_gate,
            "gate_failures":      self.gate_failures,
        }

    def summary(self) -> str:
        gate_str = "PASS" if self.passed_gate else f"FAIL ({', '.join(self.gate_failures)})"
        return (
            f"RAGAS [{gate_str}] | n={self.sample_count} | "
            f"faithfulness={self.faithfulness:.3f} "
            f"relevancy={self.answer_relevancy:.3f} "
            f"precision={self.context_precision:.3f} "
            f"recall={self.context_recall:.3f} "
            f"correctness={self.answer_correctness:.3f} "
            f"hallucination={self.hallucination_rate:.3f}"
        )


# ── Gate check ────────────────────────────────────────────────────────────────

def _check_gate(scores: dict[str, float]) -> tuple[bool, list[str]]:
    """
    Check computed scores against hard-block thresholds.
    Returns (passed, list_of_failure_messages).
    """
    failures = []
    for metric, thresholds in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        block = thresholds["hard_block"]
        lower_better = thresholds.get("lower_is_better", False)

        if lower_better:
            if score > block:
                failures.append(f"{metric}={score:.3f} > hard_block={block}")
        else:
            if score < block:
                failures.append(f"{metric}={score:.3f} < hard_block={block}")

    return len(failures) == 0, failures


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_ragas_evaluation(
    samples: list[RAGASInput],
    llm=None,
    embeddings=None,
) -> RAGASScores:
    """
    Run RAGAS evaluation over a list of RAGASInput samples.

    Args:
        samples:    List of (question, answer, contexts, ground_truth) samples.
        llm:        Optional LLM override (uses RAGAS default if None).
        embeddings: Optional embeddings override.

    Returns:
        RAGASScores with all 6 metrics and gate result.
    """
    if not samples:
        raise ValueError("Cannot evaluate empty sample list")

    logger.info("Starting RAGAS evaluation on %d samples", len(samples))
    t0 = time.perf_counter()

    # Build HuggingFace Dataset (RAGAS native format)
    data = {
        "question":    [s.question for s in samples],
        "answer":      [s.answer for s in samples],
        "contexts":    [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    }
    dataset = Dataset.from_dict(data)

    kwargs: dict[str, Any] = {"metrics": ALL_METRICS}
    if llm:
        kwargs["llm"] = llm
    if embeddings:
        kwargs["embeddings"] = embeddings

    result = evaluate(dataset, **kwargs)
    elapsed = time.perf_counter() - t0

    # Extract per-metric means
    df = result.to_pandas()
    computed = {
        "faithfulness":       float(df["faithfulness"].mean()),
        "answer_relevancy":   float(df["answer_relevancy"].mean()),
        "context_precision":  float(df["context_precision"].mean()),
        "context_recall":     float(df["context_recall"].mean()),
        "answer_correctness": float(df["answer_correctness"].mean()),
    }
    computed["hallucination_rate"] = round(1.0 - computed["faithfulness"], 4)

    passed, gate_failures = _check_gate(computed)

    scores = RAGASScores(
        **computed,
        sample_count=len(samples),
        eval_seconds=elapsed,
        passed_gate=passed,
        gate_failures=gate_failures,
    )

    logger.info(scores.summary())
    return scores


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_scores_to_mlflow(
    scores: RAGASScores,
    run_name: str,
    experiment_name: str = "ragas-evaluation",
    extra_params: dict[str, Any] | None = None,
    cfg: EnvConfig | None = None,
) -> str:
    """
    Log RAGAS scores to MLflow.  Returns the run_id.
    """
    cfg = cfg or EnvConfig()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        params = {"sample_count": scores.sample_count, **(extra_params or {})}
        mlflow.log_params(params)

        # Metrics
        mlflow.log_metrics(scores.to_dict())

        # Tags
        mlflow.set_tags({
            "gate_passed":    str(scores.passed_gate),
            "gate_failures":  "|".join(scores.gate_failures) if scores.gate_failures else "none",
            "pipeline":       "P09-RAGAS",
        })

        run_id = run.info.run_id

    logger.info("RAGAS scores logged to MLflow run: %s", run_id)
    return run_id


# ── Compare against champion baseline ────────────────────────────────────────

def compare_to_champion(
    scores: RAGASScores,
    model_name: str,
    cfg: EnvConfig | None = None,
    required_improvement_pct: float = 2.0,
) -> tuple[bool, str]:
    """
    Compare new scores to the current champion model in MLflow Registry.

    Returns:
        (better_than_champion, reason_string)
    """
    cfg = cfg or EnvConfig()
    client = get_mlflow_client(cfg)

    try:
        champion_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not champion_versions:
            logger.info("No champion model found — new scores accepted by default")
            return True, "no_champion_baseline"

        champion_run_id = champion_versions[0].run_id
        champion_data = client.get_run(champion_run_id).data.metrics

        champion_faithfulness = champion_data.get("faithfulness", 0.0)
        new_faithfulness = scores.faithfulness
        improvement = (new_faithfulness - champion_faithfulness) / max(champion_faithfulness, 1e-6) * 100

        if improvement >= required_improvement_pct:
            reason = f"faithfulness improved {improvement:.1f}% over champion ({champion_faithfulness:.3f} → {new_faithfulness:.3f})"
            logger.info("New model beats champion: %s", reason)
            return True, reason
        elif new_faithfulness >= THRESHOLDS["faithfulness"]["hard_block"]:
            reason = f"faithfulness within acceptable range ({new_faithfulness:.3f} >= {THRESHOLDS['faithfulness']['hard_block']}), not significantly better than champion"
            logger.info("New model acceptable but not significantly better: %s", reason)
            return False, reason
        else:
            reason = f"faithfulness {new_faithfulness:.3f} below hard block {THRESHOLDS['faithfulness']['hard_block']}"
            logger.warning("New model fails hard block: %s", reason)
            return False, reason

    except Exception as exc:
        logger.exception("Champion comparison failed: %s", exc)
        return False, f"comparison_error: {exc}"


def evaluate_ragas(
    qa_pairs: list,
    top_k: int = 5,
    run_name: str = "dvc-baseline",
    log_to_mlflow: bool = True,
) -> dict:
    """Wrapper: retrieve contexts, generate answers, score with RAGAS."""
    from pipelines.components.shared.base import EnvConfig
    cfg = EnvConfig()
    ragas_inputs = []
    domain_map = {"tech": "tech_docs", "hr": "hr_policies", "org": "org_info"}
    try:
        from src.storage.qdrant.retriever import search
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg.embedding_model)
        for pair in qa_pairs:
            q  = getattr(pair, "question",     None) or pair.get("question", "")
            gt = getattr(pair, "ground_truth", None) or pair.get("ground_truth", "")
            dm = getattr(pair, "domain",       None) or pair.get("domain", "tech")
            col = domain_map.get(dm, "tech_docs")
            vec = model.encode(q, normalize_embeddings=True).tolist()
            hits = search(collection_name=col, query_vector=vec, top_k=top_k,
                          qdrant_host=cfg.qdrant_host, qdrant_port=cfg.qdrant_port)
            ctxs = [h.payload.get("text", "") for h in hits if h.payload]
            ragas_inputs.append(RAGASInput(question=q, ground_truth=gt, contexts=ctxs, answer=""))
    except Exception as exc:
        print("WARNING: Qdrant unavailable:", exc)
        for pair in qa_pairs:
            q  = getattr(pair, "question",     None) or pair.get("question", "")
            gt = getattr(pair, "ground_truth", None) or pair.get("ground_truth", "")
            ragas_inputs.append(RAGASInput(question=q, ground_truth=gt, contexts=[], answer=""))
    try:
        import httpx
        for ri in ragas_inputs:
            ctx = " ".join(ri.contexts[:3])
            prompt = "Context: " + ctx + "\n\nQuestion: " + ri.question + "\n\nAnswer:"
            resp = httpx.post(cfg.ollama_base_url + "/api/generate",
                json={"model": "mistral:7b", "prompt": prompt, "stream": False}, timeout=60)
            ri.answer = resp.json().get("response", "").strip()
    except Exception as exc:
        print("WARNING: Ollama unavailable:", exc)
        for ri in ragas_inputs:
            ri.answer = "Answer unavailable - Ollama not reachable"
    result = run_ragas_evaluation(
        inputs=ragas_inputs,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_experiment="ragas-evaluation",
        run_name=run_name,
        log_to_mlflow=log_to_mlflow,
    )
    return {
        "metrics":       result.to_dict() if hasattr(result, "to_dict") else {},
        "gate_passed":   result.gate_passed,
        "gate_failures": result.gate_failures,
        "run_id":        getattr(result, "mlflow_run_id", ""),
        "sample_count":  len(ragas_inputs),
    }
