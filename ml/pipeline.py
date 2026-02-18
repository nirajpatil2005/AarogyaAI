"""
Local-only prediction pipeline integrating all Phase 1 components.
"""

from typing import Any

import numpy as np

from llm.red_flags import RedFlagEngine
from llm.sanitizer import Sanitizer
from llm.schemas import LocalPatientRecord
from ml.feature_extractor import FeatureExtractor
from ml.model import LocalHealthModel
from ml.storage import EncryptedStorage
from ml.vector_search import VectorSearch


class LocalPredictionPipeline:
    """
    Complete local-only prediction pipeline.
    Combines red-flag detection, feature extraction, ML prediction, and vector search.
    """

    def __init__(
        self,
        model_path: str = "models/health_model.json",
        storage_path: str = "data/local.db",
        vector_index_path: str = "data/vector_index.faiss",
        storage_key: str | None = None,
    ):
        """
        Initialize local prediction pipeline.

        Args:
            model_path: Path to ML model
            storage_path: Path to encrypted database
            vector_index_path: Path to vector index
            storage_key: Encryption key for storage
        """
        # Initialize components
        self.red_flag_engine = RedFlagEngine()
        self.sanitizer = Sanitizer()
        self.feature_extractor = FeatureExtractor()
        self.model = LocalHealthModel(model_path=model_path)
        self.storage = EncryptedStorage(db_path=storage_path, key=storage_key)
        self.vector_search = VectorSearch(index_path=vector_index_path)

        # Try to load existing model
        self.model.load()

    def process_query(
        self,
        symptoms: list[str],
        vitals: dict[str, float] | None = None,
        patient_id: str | None = None,
        device_id: str = "unknown",
    ) -> dict[str, Any]:
        """
        Process a health query through the complete pipeline.

        Args:
            symptoms: List of symptom descriptions
            vitals: Optional vital signs
            patient_id: Optional patient identifier
            device_id: Device making the query

        Returns:
            Complete prediction results
        """
        # Step 1: Red-flag check (emergency detection)
        red_flag_result = self.red_flag_engine.evaluate(symptoms=symptoms, vitals=vitals or {})

        # If emergency, return immediately
        if red_flag_result.is_emergency:
            # Generate emergency recommendations
            emergency_recommendations = [
                "SEEK IMMEDIATE MEDICAL ATTENTION",
                "Call emergency services (911) or go to nearest emergency room",
                "Do not delay - this may be a medical emergency",
            ]

            # Still log the query
            self.storage.store_query(
                {
                    "device_id": device_id,
                    "patient_id": patient_id,
                    "symptoms": symptoms,
                    "vitals": vitals,
                    "urgency_level": red_flag_result.urgency_level,
                    "is_emergency": True,
                }
            )

            return {
                "status": "EMERGENCY",
                "urgency_level": red_flag_result.urgency_level,
                "is_emergency": True,
                "rationale": red_flag_result.rationale,
                "recommendations": emergency_recommendations,
                "ml_prediction": None,
                "similar_cases": [],
            }

        # Step 2: Get patient data if available
        patient_data = None
        if patient_id:
            patient_data = self.storage.get_patient(patient_id)

        # Step 3: Search for similar cases in history
        similar_cases = self.vector_search.search_symptoms(symptoms, k=3)

        # Step 4: ML prediction if we have patient data
        ml_prediction = None
        if patient_data:
            try:
                # Extract features from patient record
                patient_record = LocalPatientRecord(**patient_data)
                features = self.feature_extractor.extract(patient_record)

                # Make prediction
                prediction = self.model.predict(np.array(features.features))
                ml_prediction = prediction
            except Exception as e:
                print(f"ML prediction failed: {e}")
                ml_prediction = {"error": str(e)}

        # Step 5: Store query in history
        query_data = {
            "device_id": device_id,
            "patient_id": patient_id,
            "symptoms": symptoms,
            "vitals": vitals,
            "urgency_level": red_flag_result.urgency_level,
            "is_emergency": False,
            "predictions": ml_prediction,
        }
        self.storage.store_query(query_data)

        # Step 6: Compile recommendations
        recommendations = self._generate_recommendations(
            red_flag_result=red_flag_result,
            ml_prediction=ml_prediction,
            similar_cases=similar_cases,
        )

        return {
            "status": "SUCCESS",
            "urgency_level": red_flag_result.urgency_level,
            "is_emergency": False,
            "rationale": red_flag_result.rationale,
            "ml_prediction": ml_prediction,
            "similar_cases": similar_cases,
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        red_flag_result: Any,
        ml_prediction: dict[str, Any] | None,
        similar_cases: list[dict[str, Any]],
    ) -> list[str]:
        """Generate conservative recommendations based on all signals"""
        recommendations = []

        # Red flag recommendations (for non-emergency urgent cases)
        if red_flag_result.urgency_level == "urgent":
            recommendations.append(
                "Your symptoms require prompt medical attention. Contact a healthcare provider soon."
            )

        # ML-based recommendations
        if ml_prediction and ml_prediction.get("is_trained"):
            risk_category = ml_prediction.get("risk_category", "medium")
            confidence = ml_prediction.get("confidence", 0)

            if risk_category == "high":
                recommendations.append(
                    "Your symptoms suggest elevated health risk. Please consult a healthcare provider."
                )
            elif risk_category == "medium" and confidence > 0.6:
                recommendations.append(
                    "Monitor your symptoms closely. If they persist or worsen, seek medical attention."
                )

        # History-based recommendations
        if similar_cases:
            recommendations.append(
                f"Found {len(similar_cases)} similar cases in your history. "
                "Consider patterns from previous episodes."
            )

        # General advice
        if not recommendations or red_flag_result.urgency_level == "routine":
            recommendations.extend(
                [
                    "Rest and stay hydrated",
                    "Monitor symptoms for changes",
                    "Contact healthcare provider if symptoms persist or worsen",
                    "This is not a substitute for professional medical advice",
                ]
            )

        return recommendations

    def train_model(self, n_samples: int = 1000, save: bool = True) -> dict[str, float]:
        """
        Train the local ML model.
        In production, this would use real encrypted patient data.

        Args:
            n_samples: Number of training samples
            save: Whether to save the trained model

        Returns:
            Training metrics
        """
        from ml.model import generate_synthetic_training_data

        # Generate training data
        X, y = generate_synthetic_training_data(n_samples=n_samples)

        # Train model
        metrics = self.model.train(X, y)

        # Save if requested
        if save:
            self.model.save()

        return metrics

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get status of all pipeline components"""
        return {
            "model_trained": self.model.is_trained,
            "model_version": self.model.model_version,
            "storage_stats": {
                "patients": self.storage.get_patient("__count__") or 0,
            },
            "vector_search_stats": self.vector_search.get_stats(),
        }
