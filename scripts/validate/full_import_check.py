"""Full import check. python scripts/validate/full_import_check.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check(desc, stmt):
    try:
        exec(stmt, {}); print('  OK  ', desc); return True
    except Exception as e:
        print('  FAIL', desc); print('      ', e); return False

checks = [
    ('shared.base',               'from pipelines.components.shared.base import EnvConfig, StepResult'),
    ('storage.qdrant.indexer',     'from src.storage.qdrant.indexer import chunk_text, index_document'),
    ('storage.qdrant.retriever',   'from src.storage.qdrant.retriever import search'),
    ('storage.redis',              'from src.storage.redis.session_store import save_session, check_rate_limit'),
    ('storage.postgres',           'from src.storage.postgres.models import Base'),
    ('serving.ollama',             'from src.serving.ollama.client import OllamaClient'),
    ('serving.vllm',               'from src.serving.vllm.client import VLLMClient'),
    ('guardrails.input',           'from src.guardrails.input_validator import validate_input'),
    ('guardrails.output',          'from src.guardrails.output_validator import validate_output'),
    ('agents.router',              'from src.agents.router.workflow_router import route_query'),
    ('agents.planner',             'from src.agents.planner.planner_agent import InvestigationPlan, create_plan'),
    ('agents.executor',            'from src.agents.executor.executor_agent import ExecutionContext, execute_plan'),
    ('agents.critic',              'from src.agents.critic.critic_agent import CriticVerdict, validate_investigation'),
    ('workflows.simple_research',  'from src.workflows.simple_research.workflow import run'),
    ('workflows.react',            'from src.workflows.react.workflow import run'),
    ('workflows.smart_tools',      'from src.workflows.smart_tools.workflow import run'),
    ('workflows.pec',              'from src.workflows.planner_executor_critic.workflow import run'),
    ('ragas_eval.evaluator',       'from src.ragas_eval.evaluator import RAGASScores, run_ragas_evaluation, evaluate_ragas'),
    ('ragas_eval.dataset_builder', 'from src.ragas_eval.dataset_builder import QAPair, load_pairs, load_golden_qa_from_files'),
    ('drift.evidently',            'from src.drift.evidently.drift_monitor import DriftReport, run_drift_check'),
    ('observability.langsmith',    'from src.observability.langsmith.tracer import AgentTracer'),
    ('observability.prometheus',   'from src.observability.prometheus.metrics import REQUEST_TOTAL, record_request'),
    ('api.main',                   'from src.api.main import app'),
    ('pipelines.p05',              'from pipelines.p05_data_indexing.pipeline import data_indexing_pipeline'),
    ('pipelines.p09',              'from pipelines.p09_ragas_evaluation.pipeline import ragas_evaluation_pipeline'),
    ('pipelines.p11',              'from pipelines.p11_automated_retraining.pipeline import automated_retraining_pipeline'),
    ('mlops.model_registry',       'from mlops.model_registry.promote_to_champion import promote_to_champion'),
    ('mlops.lineage',              'from mlops.lineage.lineage_tracker import trace_from_mlflow_run'),
]

print(f'Running {len(checks)} import checks...')
print()
passed = sum(check(d, s) for d, s in checks)
failed = len(checks) - passed
print()
print('=' * 50)
print(f'  {passed}/{len(checks)} PASSED   {failed} FAILED')
print('=' * 50)
sys.exit(0 if failed == 0 else 1)
