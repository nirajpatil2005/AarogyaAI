"""
Feature extractor for ML model.
Converts patient data into feature vectors for prediction.
"""

import numpy as np

from llm.schemas import LocalPatientRecord, MLFeatureVector


class FeatureExtractor:
    """
    Extract features from patient records for ML models.

    Features include:
    - Demographics (age, sex)
    - Condition flags (one-hot encoded)
    - Lab value statistics
    - Symptom patterns
    """

    KNOWN_CONDITIONS = [
        "hypertension",
        "diabetes",
        "asthma",
        "copd",
        "heart_disease",
        "obesity",
        "anxiety",
        "depression",
    ]

    def __init__(self):
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self) -> list[str]:
        """Generate list of feature names"""
        names = ["age", "sex_m", "sex_f"]

        # Condition flags
        for condition in self.KNOWN_CONDITIONS:
            names.append(f"has_{condition}")

        # Lab value slots
        for i in range(5):
            names.append(f"lab_value_{i}")

        return names

    def extract(self, patient: LocalPatientRecord) -> MLFeatureVector:
        """
        Extract feature vector from patient record.

        Args:
            patient: Patient record

        Returns:
            MLFeatureVector ready for model input
        """
        features = []

        # Age
        features.append(float(patient.age))

        # Sex (one-hot)
        features.append(1.0 if patient.sex.value == "M" else 0.0)
        features.append(1.0 if patient.sex.value == "F" else 0.0)

        # Condition flags
        patient_conditions_lower = [c.lower() for c in patient.conditions]
        for condition in self.KNOWN_CONDITIONS:
            has_condition = 1.0 if condition in patient_conditions_lower else 0.0
            features.append(has_condition)

        # Recent lab values (last 5)
        recent_labs = patient.lab_results[-5:] if len(patient.lab_results) > 0 else []
        for i in range(5):
            if i < len(recent_labs):
                # Normalize lab value (simplified)
                features.append(recent_labs[i].value / 100.0)
            else:
                features.append(0.0)

        return MLFeatureVector(
            features=features, feature_names=self.feature_names, metadata={"version": "0.1.0"}
        )


# Stub for local ML model
class LocalHealthModel:
    """
    Stub for local ML model (XGBoost).
    Phase 1 will implement actual training and prediction.
    """

    def __init__(self):
        self.model = None  # Will be XGBoost model
        self.is_trained = False

    def predict(self, features: MLFeatureVector) -> dict[str, float]:
        """
        Predict health risk scores.

        Returns dict of risk scores for various conditions.
        """
        # Stub implementation
        return {"general_health_risk": 0.3, "cardiovascular_risk": 0.2, "respiratory_risk": 0.1}

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train model on local data.
        Phase 1 implementation.
        """
        # Stub - will implement in Phase 1
        pass
