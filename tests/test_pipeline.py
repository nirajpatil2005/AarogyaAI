"""
Tests for local prediction pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ml.pipeline import LocalPredictionPipeline


@pytest.fixture
def temp_dir():
    """Create temporary directory for pipeline data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def pipeline(temp_dir):
    """Create pipeline instance for testing"""
    # Mock sentence transformer to avoid network calls
    with patch("ml.vector_search.SentenceTransformer") as mock_st:
        import numpy as np
        from unittest.mock import Mock

        mock_encoder = Mock()
        mock_encoder.get_sentence_embedding_dimension.return_value = 384
        mock_encoder.encode = lambda texts, **kwargs: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        ).astype("float32")
        mock_st.return_value = mock_encoder

        pipeline = LocalPredictionPipeline(
            model_path=str(Path(temp_dir) / "model.json"),
            storage_path=str(Path(temp_dir) / "storage.db"),
            vector_index_path=str(Path(temp_dir) / "index.faiss"),
            storage_key="test-key",
        )
        yield pipeline


class TestLocalPredictionPipeline:
    """Test suite for LocalPredictionPipeline"""

    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.red_flag_engine is not None
        assert pipeline.sanitizer is not None
        assert pipeline.feature_extractor is not None
        assert pipeline.model is not None
        assert pipeline.storage is not None
        assert pipeline.vector_search is not None

    def test_process_emergency_query(self, pipeline):
        """Test processing emergency symptoms"""
        symptoms = ["severe chest pain", "shortness of breath"]
        result = pipeline.process_query(symptoms, device_id="test-001")

        assert result["status"] == "EMERGENCY"
        assert result["is_emergency"] is True
        assert result["urgency_level"] == "immediate"
        assert len(result["recommendations"]) > 0

    def test_process_routine_query(self, pipeline):
        """Test processing routine symptoms"""
        symptoms = ["mild headache", "slight fatigue"]
        result = pipeline.process_query(symptoms, device_id="test-002")

        assert result["status"] == "SUCCESS"
        assert result["is_emergency"] is False
        assert "recommendations" in result

    def test_process_query_with_vitals(self, pipeline):
        """Test processing with vital signs"""
        symptoms = ["fever", "cough"]
        vitals = {"temperature": 100.5, "heart_rate": 85}
        result = pipeline.process_query(symptoms, vitals=vitals, device_id="test-003")

        assert result is not None
        assert "urgency_level" in result

    def test_train_model(self, pipeline):
        """Test model training"""
        metrics = pipeline.train_model(n_samples=100, save=False)

        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert pipeline.model.is_trained is True

    def test_get_pipeline_status(self, pipeline):
        """Test getting pipeline status"""
        status = pipeline.get_pipeline_status()

        assert "model_trained" in status
        assert "model_version" in status
        assert "storage_stats" in status
        assert "vector_search_stats" in status
