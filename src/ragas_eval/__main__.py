"""
CLI entrypoint for DVC ragas pipeline stages.
Called by dvc.yaml as: python -m src.ragas_eval.evaluator --mode <stage>

Modes:
  validate   --data-dir data/golden_qa --output data/golden_qa/coverage_report.json
  baseline   --golden-qa-dir data/golden_qa --output data/ragas_results/baseline.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run_validate(data_dir: str, output: str) -> dict:
    """Check golden QA coverage per domain — must have >= 20 pairs each."""
    data_path = Path(data_dir)
    required_domains = ["tech", "hr", "org"]
    MIN_PAIRS = 20

    coverage = {}
    failures = []

    for domain in required_domains:
        qa_file = data_path / domain / "qa_pairs.json"
        if not qa_file.exists():
            coverage[domain] = 0
            failures.append(f"{domain}: file missing")
            continue
        pairs = json.loads(qa_file.read_text(encoding="utf-8"))
        active = [p for p in pairs if p.get("active", True)]
        coverage[domain] = len(active)
        if len(active) < MIN_PAIRS:
            failures.append(f"{domain}: only {len(active)} pairs (need {MIN_PAIRS})")

    report = {
        "coverage":     coverage,
        "min_required": MIN_PAIRS,
        "failures":     failures,
        "passed":       len(failures) == 0,
        "total_pairs":  sum(coverage.values()),
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "PASS" if report["passed"] else f"FAIL: {failures}"
    print(f"QA Coverage [{status}]: {coverage}")

    if not report["passed"]:
        print("ERROR: Insufficient golden QA coverage. Run: python scripts/data/seed_golden_qa.py")
        sys.exit(1)

    return report


def run_baseline(golden_qa_dir: str, output: str, params_file: str) -> dict:
    """
    Run RAGAS baseline evaluation against golden QA dataset.
    Requires Qdrant and Ollama to be running.
    """
    import yaml
    from src.ragas_eval.evaluator import evaluate_ragas
    from src.ragas_eval.dataset_builder import load_golden_qa_from_files

    params = {}
    if params_file and os.path.exists(params_file):
        with open(params_file) as f:
            params = yaml.safe_load(f) or {}

    top_k = params.get("rag", {}).get("top_k", 5)

    print(f"Loading golden QA pairs from {golden_qa_dir}...")
    qa_pairs = load_golden_qa_from_files(golden_qa_dir)
    print(f"Loaded {len(qa_pairs)} QA pairs across all domains")

    print("Running RAGAS evaluation (requires Qdrant + Ollama)...")
    result = evaluate_ragas(
        qa_pairs=qa_pairs,
        top_k=top_k,
        run_name="dvc-baseline",
        log_to_mlflow=True,
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("RAGAS Baseline Results:")
    for metric, val in result.get("metrics", {}).items():
        print(f"  {metric:<24} {val:.4f}")
    print(f"Gate passed: {result.get('gate_passed', False)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="RAGAS DVC stage runner")
    parser.add_argument("--mode",           required=True, choices=["validate", "baseline"])
    parser.add_argument("--data-dir",       default="data/golden_qa")
    parser.add_argument("--golden-qa-dir",  default="data/golden_qa")
    parser.add_argument("--output",         default="data/ragas_results/baseline.json")
    parser.add_argument("--params",         default="mlops/dvc/params/rag_params.yaml")
    args = parser.parse_args()

    print(f"Running ragas stage: {args.mode}")

    if args.mode == "validate":
        run_validate(args.data_dir, args.output)
    elif args.mode == "baseline":
        run_baseline(args.golden_qa_dir, args.output, args.params)


if __name__ == "__main__":
    main()
