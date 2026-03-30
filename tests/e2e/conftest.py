"""
tests/e2e/conftest.py
pytest configuration for E2E test suite.
"""
import pytest

def pytest_addoption(parser):
    parser.addoption("--vllm-url",      default="http://localhost:8000")
    parser.addoption("--qdrant-host",   default="localhost")
    parser.addoption("--mlflow-uri",    default="http://localhost:5000")
    parser.addoption("--postgres-host", default="localhost")

def pytest_configure(config):
    config.addinivalue_line("markers", "l1_qdrant: Qdrant retrieval tests")
    config.addinivalue_line("markers", "l2_vllm: vLLM inference latency tests")
    config.addinivalue_line("markers", "l3_rag: Full RAG flow tests")
    config.addinivalue_line("markers", "l4_ragas: RAGAS spot-check tests")
    config.addinivalue_line("markers", "l5_dvc: DVC / Qdrant point count tests")
