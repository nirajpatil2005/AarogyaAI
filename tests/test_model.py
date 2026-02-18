"""
Tests for local XGBoost model.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.model import LocalHealthModel, generate_synthetic_training_data


@pytest.fixture
def temp_model_path():
    """Create temporary path for model"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_model.json"


@pytest.fixture
def trained_model(temp_model_path):
    """Create and train a model for testing"""
    model = LocalHealthModel(model_path=str(temp_model_path))
    X, y = generate_synthetic_training_data(n_samples=200)
    model.train(X, y)
    return model


class TestLocalHealthModel:
    """Test suite for LocalHealthModel"""

    def test_initialization(self, temp_model_path):
        """Test model initialization"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        assert model.model is None
        assert model.is_trained is False
        assert model.model_version == "1.0.0"
        assert len(model.risk_categories) == 3

    def test_training(self, temp_model_path):
        """Test model training"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        X, y = generate_synthetic_training_data(n_samples=200)

        metrics = model.train(X, y)

        assert model.is_trained is True
        assert model.model is not None
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.3  # Should do better than random
        assert metrics["n_samples"] == 200

    def test_prediction_untrained(self, temp_model_path):
        """Test prediction on untrained model returns default"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        features = np.random.rand(16)  # 16 features

        result = model.predict(features)

        assert result["is_trained"] is False
        assert result["risk_category"] in ["low", "medium", "high"]
        assert 0 <= result["risk_score"] <= 1
        assert "probabilities" in result

    def test_prediction_trained(self, trained_model):
        """Test prediction on trained model"""
        features = np.random.rand(16)

        result = trained_model.predict(features)

        assert result["is_trained"] is True
        assert result["risk_category"] in ["low", "medium", "high"]
        assert 0 <= result["risk_score"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result
        assert len(result["probabilities"]) == 3

    def test_prediction_1d_features(self, trained_model):
        """Test prediction with 1D feature array"""
        features = np.random.rand(16)
        result = trained_model.predict(features)
        assert result is not None

    def test_prediction_2d_features(self, trained_model):
        """Test prediction with 2D feature array"""
        features = np.random.rand(1, 16)
        result = trained_model.predict(features)
        assert result is not None

    def test_batch_prediction(self, trained_model):
        """Test batch prediction"""
        features = np.random.rand(5, 16)
        results = trained_model.predict_batch(features)

        assert len(results) == 5
        for result in results:
            assert "risk_category" in result
            assert "risk_score" in result

    def test_save_and_load(self, temp_model_path):
        """Test model saving and loading"""
        # Train and save model
        model1 = LocalHealthModel(model_path=str(temp_model_path))
        X, y = generate_synthetic_training_data(n_samples=200)
        model1.train(X, y)
        
        # Make prediction with original model
        test_features = np.random.rand(16)
        pred1 = model1.predict(test_features)

        # Save model
        success = model1.save()
        assert success is True
        assert temp_model_path.exists()

        # Load model
        model2 = LocalHealthModel(model_path=str(temp_model_path))
        success = model2.load()
        assert success is True
        assert model2.is_trained is True

        # Make prediction with loaded model
        pred2 = model2.predict(test_features)

        # Predictions should be identical
        assert pred1["risk_category"] == pred2["risk_category"]
        assert abs(pred1["risk_score"] - pred2["risk_score"]) < 0.01

    def test_save_untrained_model(self, temp_model_path):
        """Test that saving untrained model fails gracefully"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        success = model.save()
        assert success is False

    def test_load_nonexistent_model(self, temp_model_path):
        """Test loading non-existent model"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        success = model.load()
        assert success is False

    def test_feature_importance(self, trained_model):
        """Test feature importance extraction"""
        importance = trained_model.get_feature_importance()

        assert len(importance) > 0
        assert all(isinstance(v, float) for v in importance.values())
        # All importance scores should be non-negative
        assert all(v >= 0 for v in importance.values())

    def test_untrained_feature_importance(self, temp_model_path):
        """Test feature importance on untrained model"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        importance = model.get_feature_importance()
        assert importance == {}

    def test_training_with_custom_params(self, temp_model_path):
        """Test training with custom parameters"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        X, y = generate_synthetic_training_data(n_samples=200)

        custom_params = {"max_depth": 4, "learning_rate": 0.05}
        metrics = model.train(X, y, params=custom_params)

        assert model.is_trained is True
        assert metrics["train_accuracy"] > 0

    def test_training_validation_split(self, temp_model_path):
        """Test training with different validation split"""
        model = LocalHealthModel(model_path=str(temp_model_path))
        X, y = generate_synthetic_training_data(n_samples=200)

        metrics = model.train(X, y, validation_split=0.3)

        assert model.is_trained is True
        assert metrics["n_samples"] == 200


class TestSyntheticDataGeneration:
    """Test suite for synthetic data generation"""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        X, y = generate_synthetic_training_data(n_samples=100)

        assert X.shape[0] == 100
        assert X.shape[1] == 16  # Expected number of features
        assert y.shape[0] == 100
        assert set(np.unique(y)).issubset({0, 1, 2})  # Valid risk categories

    def test_synthetic_data_distribution(self):
        """Test that synthetic data has reasonable distribution"""
        X, y = generate_synthetic_training_data(n_samples=1000)

        # Age should be in reasonable range
        ages = X[:, 0]
        assert ages.min() >= 18
        assert ages.max() <= 90

        # Sex should be binary
        assert set(np.unique(X[:, 1])).issubset({0, 1})

        # Should have at least 2 risk categories (sometimes may not have all 3)
        assert len(np.unique(y)) >= 2
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible"""
        X1, y1 = generate_synthetic_training_data(n_samples=100)
        X2, y2 = generate_synthetic_training_data(n_samples=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
