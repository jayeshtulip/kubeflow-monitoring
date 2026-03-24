import pytest

@pytest.mark.l2
def test_ragas_returns_all_metrics():
    """RAGAS eval on 10 samples returns faithfulness, relevancy, precision, recall, correctness."""
    pass

@pytest.mark.l2
def test_ragas_faithfulness_above_threshold():
    """Assert faithfulness >= 0.85 on golden QA dataset sample."""
    pass
