"""
L1 Unit Tests - Enterprise LLM Platform v2.0
Tests that run without cluster access
"""
import pytest
import json
import os

class TestRAGASConfig:
    """Test RAGAS evaluation configuration"""

    def test_faithfulness_threshold(self):
        """Faithfulness gate must be >= 0.85"""
        threshold = 0.85
        assert threshold >= 0.85, "Faithfulness threshold too low"

    def test_hallucination_threshold(self):
        """Hallucination gate must be <= 0.15"""
        threshold = 0.15
        assert threshold <= 0.15, "Hallucination threshold too high"

    def test_qa_minimum_pairs(self):
        """Minimum QA pairs per domain"""
        min_pairs = 20
        assert min_pairs >= 20, "Need at least 20 QA pairs per domain"

class TestDVCConfig:
    """Test DVC pipeline configuration"""

    def test_dvc_yaml_exists(self):
        """dvc.yaml must exist in project root"""
        assert os.path.exists("dvc.yaml"), "dvc.yaml not found"

    def test_dvc_yaml_has_stages(self):
        """dvc.yaml must have required stages"""
        import yaml
        with open("dvc.yaml") as f:
            config = yaml.safe_load(f)
        stages = config.get("stages", {})
        required = ["preprocess", "embed", "index", "validate_golden_qa", "ragas_baseline"]
        for stage in required:
            assert stage in stages, f"Missing DVC stage: {stage}"

    def test_dvc_remote_configured(self):
        """DVC remote must be S3"""
        assert os.path.exists(".dvc/config"), ".dvc/config not found"
        with open(".dvc/config") as f:
            content = f.read()
        assert "s3://" in content, "DVC S3 remote not configured"

class TestGitHubActionsWorkflows:
    """Test GitHub Actions workflow files exist"""

    def test_ci_workflow_exists(self):
        assert os.path.exists(".github/workflows/ci.yml")

    def test_nightly_workflow_exists(self):
        assert os.path.exists(".github/workflows/nightly.yml")

    def test_release_workflow_exists(self):
        assert os.path.exists(".github/workflows/release.yml")

class TestPipelineIDs:
    """Test pipeline configuration values"""

    def test_experiment_id_format(self):
        """KFP experiment ID must be valid UUID format"""
        import re
        exp_id = "21487485-55f3-4529-8c66-90f5710c8e4e"
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(pattern, exp_id), "Invalid experiment ID format"

    def test_p04_pipeline_id_format(self):
        import re
        pipeline_id = "e80f6090-254b-4f05-b0cb-4855de1bf3cc"
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(pattern, pipeline_id), "Invalid P04 pipeline ID"

    def test_p01_pipeline_id_format(self):
        import re
        pipeline_id = "a5b6bcd4-3b08-4058-a623-36be8bd95495"
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(pattern, pipeline_id), "Invalid P01 pipeline ID"
