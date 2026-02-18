"""
Local XGBoost model for health risk prediction.
Trains on local data and makes predictions without cloud dependencies.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from ml.feature_extractor import FeatureExtractor


class LocalHealthModel:
    """
    Local XGBoost model for health risk prediction.
    Trains on encrypted local data and makes privacy-preserving predictions.
    """

    def __init__(self, model_path: str = "models/health_model.json"):
        """
        Initialize local health model.

        Args:
            model_path: Path to save/load model
        """
        self.model_path = model_path
        self.model: xgb.XGBClassifier | None = None
        self.is_trained = False
        self.feature_extractor = FeatureExtractor()
        self.model_version = "1.0.0"
        self.risk_categories = ["low", "medium", "high"]

        # Create models directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Train XGBoost model on local data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            validation_split: Fraction of data for validation
            params: XGBoost hyperparameters

        Returns:
            Training metrics
        """
        # Default parameters optimized for health data
        default_params = {
            "objective": "multi:softmax",
            "num_class": 3,  # low, medium, high risk
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "mlogloss",
        }

        if params:
            default_params.update(params)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Initialize and train model
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.is_trained = True

        # Calculate metrics
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        metrics = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "n_samples": len(X),
            "n_features": X.shape[1],
        }

        return metrics

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """
        Predict health risk scores.

        Args:
            features: Feature vector or matrix

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_trained or self.model is None:
            # Return default low-confidence prediction if not trained
            return {
                "risk_category": "medium",
                "risk_score": 0.5,
                "confidence": 0.3,
                "probabilities": {"low": 0.33, "medium": 0.34, "high": 0.33},
                "is_trained": False,
            }

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get predictions and probabilities
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        # Convert to risk category
        risk_category = self.risk_categories[int(predictions[0])]
        risk_score = float(probabilities[0][int(predictions[0])])

        # Confidence is the difference between top 2 probabilities
        sorted_probs = np.sort(probabilities[0])[::-1]
        confidence = float(sorted_probs[0] - sorted_probs[1])

        return {
            "risk_category": risk_category,
            "risk_score": risk_score,
            "confidence": confidence,
            "probabilities": {
                "low": float(probabilities[0][0]),
                "medium": float(probabilities[0][1]),
                "high": float(probabilities[0][2]),
            },
            "is_trained": True,
        }

    def predict_batch(self, features: np.ndarray) -> list[dict[str, Any]]:
        """
        Predict for multiple samples.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            List of prediction dictionaries
        """
        results = []
        for i in range(features.shape[0]):
            result = self.predict(features[i])
            results.append(result)
        return results

    def save(self, path: str | None = None) -> bool:
        """
        Save model to disk.

        Args:
            path: Optional custom path (uses self.model_path if None)

        Returns:
            True if successful
        """
        if not self.is_trained or self.model is None:
            print("Cannot save untrained model")
            return False

        save_path = path or self.model_path

        try:
            # Save model in XGBoost JSON format (human-readable)
            self.model.save_model(save_path)

            # Save metadata
            metadata_path = str(save_path).replace(".json", "_metadata.json")
            metadata = {
                "version": self.model_version,
                "feature_names": self.feature_extractor.feature_names,
                "risk_categories": self.risk_categories,
                "is_trained": self.is_trained,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load(self, path: str | None = None) -> bool:
        """
        Load model from disk.

        Args:
            path: Optional custom path (uses self.model_path if None)

        Returns:
            True if successful
        """
        load_path = path or self.model_path

        if not Path(load_path).exists():
            print(f"Model file not found: {load_path}")
            return False

        try:
            # Load model
            self.model = xgb.XGBClassifier()
            self.model.load_model(load_path)
            self.is_trained = True

            # Load metadata if available
            metadata_path = str(load_path).replace(".json", "_metadata.json")
            if Path(metadata_path).exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get("version", self.model_version)
                    self.risk_categories = metadata.get("risk_categories", self.risk_categories)

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        importance_scores = self.model.feature_importances_
        feature_names = self.feature_extractor.feature_names

        return {
            name: float(score) for name, score in zip(feature_names, importance_scores)
        }


def generate_synthetic_training_data(n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for model development.
    In production, this would use real encrypted patient data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        (X, y) tuple of features and labels
    """
    np.random.seed(42)

    extractor = FeatureExtractor()
    n_features = len(extractor.feature_names)

    # Generate features
    X = np.random.rand(n_samples, n_features)

    # Age (18-90)
    X[:, 0] = np.random.randint(18, 90, n_samples)

    # Sex (binary)
    X[:, 1] = np.random.randint(0, 2, n_samples)
    X[:, 2] = 1 - X[:, 1]

    # Conditions (sparse binary)
    X[:, 3:11] = (np.random.rand(n_samples, 8) > 0.7).astype(float)

    # Lab values (0-1 normalized)
    X[:, 11:] = np.random.rand(n_samples, 5)

    # Generate risk labels based on features
    # Higher age, more conditions, abnormal labs -> higher risk
    risk_score = (
        X[:, 0] / 90.0 * 0.3  # Age contribution
        + X[:, 3:11].sum(axis=1) / 8.0 * 0.4  # Conditions contribution
        + np.abs(X[:, 11:] - 0.5).mean(axis=1) * 0.3  # Lab abnormality
    )

    # Convert to categories (0=low, 1=medium, 2=high)
    y = np.zeros(n_samples, dtype=int)
    y[risk_score > 0.33] = 1  # medium
    y[risk_score > 0.66] = 2  # high

    return X, y
