"""
Tests for encrypted storage module.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ml.storage import EncryptedStorage


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
def storage(temp_db):
    """Create storage instance for testing"""
    return EncryptedStorage(db_path=temp_db, key="test-key-12345")


class TestEncryptedStorage:
    """Test suite for EncryptedStorage"""

    def test_initialization(self, temp_db):
        """Test storage initialization"""
        storage = EncryptedStorage(db_path=temp_db, key="test-key")
        assert Path(temp_db).parent.exists()

    def test_store_and_retrieve_patient(self, storage):
        """Test storing and retrieving patient record"""
        patient_data = {
            "patient_id": "P001",
            "age": 45,
            "sex": "M",
            "name": "John Doe",
            "date_of_birth": "1979-01-15",
            "conditions": ["hypertension", "diabetes"],
            "medications": [{"name": "metformin", "dosage": "500mg"}],
            "allergies": ["penicillin"],
        }

        # Store patient
        success = storage.store_patient(patient_data)
        assert success is True

        # Retrieve patient
        retrieved = storage.get_patient("P001")
        assert retrieved is not None
        assert retrieved["patient_id"] == "P001"
        assert retrieved["age"] == 45
        assert retrieved["sex"] == "M"
        assert retrieved["name"] == "John Doe"
        assert "hypertension" in retrieved["conditions"]
        assert "diabetes" in retrieved["conditions"]

    def test_update_patient(self, storage):
        """Test updating existing patient record"""
        # Store initial record
        patient_data = {
            "patient_id": "P002",
            "age": 30,
            "sex": "F",
            "conditions": ["asthma"],
        }
        storage.store_patient(patient_data)

        # Update with new data
        updated_data = {
            "patient_id": "P002",
            "age": 31,
            "sex": "F",
            "conditions": ["asthma", "allergies"],
        }
        storage.store_patient(updated_data)

        # Retrieve and verify
        retrieved = storage.get_patient("P002")
        assert retrieved["age"] == 31
        assert len(retrieved["conditions"]) == 2

    def test_get_nonexistent_patient(self, storage):
        """Test retrieving non-existent patient"""
        result = storage.get_patient("NONEXISTENT")
        assert result is None

    def test_store_and_retrieve_query(self, storage):
        """Test storing and retrieving query history"""
        query_data = {
            "device_id": "dev-001",
            "patient_id": "P001",
            "symptoms": ["headache", "fever"],
            "vitals": {"temperature": 101.5, "heart_rate": 88},
            "urgency_level": "routine",
            "is_emergency": False,
            "predictions": {"general_health_risk": 0.3},
            "fingerprint": "abc123def456",
        }

        # Store query
        query_id = storage.store_query(query_data)
        assert query_id is not None

        # Retrieve query history
        history = storage.get_query_history(device_id="dev-001")
        assert len(history) == 1
        assert history[0]["device_id"] == "dev-001"
        assert "headache" in history[0]["symptoms"]
        assert history[0]["is_emergency"] is False

    def test_query_history_filtering(self, storage):
        """Test query history filtering"""
        # Store multiple queries
        for i in range(5):
            storage.store_query(
                {
                    "device_id": f"dev-{i % 2}",
                    "patient_id": f"P{i % 3}",
                    "symptoms": [f"symptom-{i}"],
                    "urgency_level": "routine",
                }
            )

        # Filter by device_id
        dev0_history = storage.get_query_history(device_id="dev-0")
        assert len(dev0_history) == 3

        # Filter by patient_id
        p1_history = storage.get_query_history(patient_id="P1")
        assert len(p1_history) == 2

        # Test limit
        limited = storage.get_query_history(limit=2)
        assert len(limited) == 2

    def test_store_and_retrieve_lab_results(self, storage):
        """Test storing and retrieving lab results"""
        # First store a patient
        storage.store_patient({"patient_id": "P003", "age": 50, "sex": "M"})

        # Store lab result
        lab_data = {
            "patient_id": "P003",
            "test_name": "Glucose",
            "value": 95.0,
            "unit": "mg/dL",
            "reference_range": "70-100",
        }
        success = storage.store_lab_result(lab_data)
        assert success is True

        # Retrieve lab results
        results = storage.get_lab_results("P003")
        assert len(results) == 1
        assert results[0]["test_name"] == "Glucose"
        assert results[0]["value"] == 95.0
        assert results[0]["unit"] == "mg/dL"

    def test_multiple_lab_results(self, storage):
        """Test storing multiple lab results"""
        storage.store_patient({"patient_id": "P004", "age": 35, "sex": "F"})

        # Store multiple lab results
        labs = [
            {"patient_id": "P004", "test_name": "Hemoglobin", "value": 13.5, "unit": "g/dL"},
            {"patient_id": "P004", "test_name": "WBC", "value": 7.2, "unit": "K/uL"},
            {"patient_id": "P004", "test_name": "Platelets", "value": 250.0, "unit": "K/uL"},
        ]

        for lab in labs:
            storage.store_lab_result(lab)

        # Retrieve and verify
        results = storage.get_lab_results("P004")
        assert len(results) == 3
        test_names = [r["test_name"] for r in results]
        assert "Hemoglobin" in test_names
        assert "WBC" in test_names
        assert "Platelets" in test_names

    def test_lab_results_limit(self, storage):
        """Test lab results limit parameter"""
        storage.store_patient({"patient_id": "P005", "age": 40, "sex": "M"})

        # Store 10 lab results
        for i in range(10):
            storage.store_lab_result(
                {
                    "patient_id": "P005",
                    "test_name": f"Test-{i}",
                    "value": float(i),
                    "unit": "unit",
                }
            )

        # Retrieve with limit
        results = storage.get_lab_results("P005", limit=5)
        assert len(results) == 5

    def test_encryption_with_different_keys(self, temp_db):
        """Test that different keys produce different encrypted data"""
        # Store data with first key
        storage1 = EncryptedStorage(db_path=temp_db, key="key1")
        storage1.store_patient({"patient_id": "P006", "age": 25, "sex": "F"})

        # Close first storage properly
        del storage1

        # Try to initialize with wrong key - should fail
        # When using wrong encryption key, sqlcipher raises MemoryError during init
        # This confirms encryption is working properly
        with pytest.raises(MemoryError):
            storage2 = EncryptedStorage(db_path=temp_db, key="key2")

    def test_clear_all_data(self, storage):
        """Test clearing all data"""
        # Add some data
        storage.store_patient({"patient_id": "P007", "age": 30, "sex": "M"})
        storage.store_query(
            {"device_id": "dev-001", "symptoms": ["test"], "urgency_level": "routine"}
        )

        # Clear all data
        storage.clear_all_data()

        # Verify everything is cleared
        assert storage.get_patient("P007") is None
        assert len(storage.get_query_history()) == 0

    def test_query_with_emergency_flag(self, storage):
        """Test storing emergency queries"""
        emergency_query = {
            "device_id": "dev-emergency",
            "symptoms": ["severe chest pain"],
            "urgency_level": "immediate",
            "is_emergency": True,
            "fingerprint": "emergency123",
        }

        storage.store_query(emergency_query)

        history = storage.get_query_history(device_id="dev-emergency")
        assert len(history) == 1
        assert history[0]["is_emergency"] is True
        assert history[0]["urgency_level"] == "immediate"

    def test_timestamp_handling(self, storage):
        """Test automatic timestamp handling"""
        query_data = {"device_id": "dev-ts", "symptoms": ["test"], "urgency_level": "routine"}

        # Store without explicit timestamp
        storage.store_query(query_data)

        # Retrieve and verify timestamp was added
        history = storage.get_query_history(device_id="dev-ts")
        assert len(history) == 1
        assert history[0]["timestamp"] is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(history[0]["timestamp"])
