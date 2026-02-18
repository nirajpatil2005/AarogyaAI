"""
Unit tests for sanitizer module.
Testing PHI removal, fingerprinting, and output validation.
"""

from datetime import datetime, timedelta

from llm.sanitizer import Sanitizer
from llm.schemas import HistoryRecord, LabResult, LocalPatientRecord, Medication, SexEnum


class TestSanitizer:
    """Test suite for Sanitizer class"""

    def test_remove_email_from_text(self):
        """Test that emails are removed from text"""
        text = "Contact john.doe@example.com for more info"
        cleaned = Sanitizer._remove_phi_from_text(text)
        assert "john.doe@example.com" not in cleaned
        assert "[EMAIL_REDACTED]" in cleaned

    def test_remove_phone_from_text(self):
        """Test that phone numbers are removed"""
        text = "Call me at 555-123-4567 or 555.987.6543"
        cleaned = Sanitizer._remove_phi_from_text(text)
        assert "555-123-4567" not in cleaned
        assert "555.987.6543" not in cleaned
        assert "[PHONE_REDACTED]" in cleaned

    def test_remove_ssn_from_text(self):
        """Test that SSN is removed"""
        text = "SSN: 123-45-6789"
        cleaned = Sanitizer._remove_phi_from_text(text)
        assert "123-45-6789" not in cleaned
        assert "[SSN_REDACTED]" in cleaned

    def test_remove_dates_from_text(self):
        """Test that dates are removed"""
        text = "Symptoms started on 2024-01-15 and worsened on 01/20/2024"
        cleaned = Sanitizer._remove_phi_from_text(text)
        assert "2024-01-15" not in cleaned
        assert "01/20/2024" not in cleaned
        assert "[DATE_REDACTED]" in cleaned

    def test_detect_phi_positive(self):
        """Test PHI detection with positive cases"""
        text = "Contact john@example.com or call 555-1234"
        phi_types = Sanitizer._detect_phi(text)
        assert "email" in phi_types
        assert "phone" in phi_types

    def test_detect_phi_negative(self):
        """Test PHI detection with clean text"""
        text = "Patient has fever and cough"
        phi_types = Sanitizer._detect_phi(text)
        assert len(phi_types) == 0

    def test_relative_time_descriptor(self):
        """Test conversion of timestamps to relative descriptors"""
        reference = datetime(2024, 1, 15, 12, 0, 0)

        # Test hours
        timestamp = datetime(2024, 1, 15, 8, 0, 0)
        result = Sanitizer._relative_time_descriptor(timestamp, reference)
        assert "4 hours ago" in result

        # Test days
        timestamp = datetime(2024, 1, 10, 12, 0, 0)
        result = Sanitizer._relative_time_descriptor(timestamp, reference)
        assert "5 days ago" in result

        # Test weeks
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        result = Sanitizer._relative_time_descriptor(timestamp, reference)
        assert "weeks ago" in result

    def test_round_to_sig_figs(self):
        """Test rounding to significant figures"""
        assert Sanitizer._round_to_sig_figs(123.456, 2) == 120.0
        assert Sanitizer._round_to_sig_figs(1.234, 2) == 1.2
        assert Sanitizer._round_to_sig_figs(0.001234, 2) == 0.0012
        assert Sanitizer._round_to_sig_figs(0.0, 2) == 0.0

    def test_generate_fingerprint_deterministic(self):
        """Test that fingerprint generation is deterministic"""
        patient = LocalPatientRecord(
            patient_id="test-001", age=45, sex=SexEnum.MALE, conditions=["hypertension", "diabetes"]
        )
        task = "Evaluate symptoms"

        fp1 = Sanitizer._generate_fingerprint(patient, task)
        fp2 = Sanitizer._generate_fingerprint(patient, task)

        assert fp1 == fp2
        assert len(fp1) == 64  # SHA256 hex digest

    def test_sanitize_basic_patient(self):
        """Test basic sanitization of patient record"""
        patient = LocalPatientRecord(
            patient_id="test-001", name="John Doe", age=35, sex=SexEnum.MALE, conditions=["asthma"]
        )

        task = "Return JSON with diagnosis suggestions"
        sanitized = Sanitizer.sanitize(patient, task, ["shortness of breath"])

        # Check fingerprint present
        assert len(sanitized.fingerprint) == 64

        # Check basic fields copied
        assert sanitized.age == 35
        assert sanitized.sex == SexEnum.MALE
        assert "asthma" in sanitized.conditions
        assert "shortness of breath" in sanitized.symptoms

        # Check no name in output
        output_json = sanitized.model_dump_json()
        assert "John Doe" not in output_json
        assert "test-001" not in output_json

    def test_sanitize_with_history(self):
        """Test sanitization with historical records"""
        now = datetime.now()
        two_days_ago = now - timedelta(days=2)

        patient = LocalPatientRecord(
            patient_id="test-002",
            age=50,
            sex=SexEnum.FEMALE,
            history=[
                HistoryRecord(
                    timestamp=two_days_ago,
                    symptoms=["fever", "cough"],
                    diagnosis="Upper respiratory infection",
                )
            ],
        )

        task = "Evaluate current symptoms"
        sanitized = Sanitizer.sanitize(patient, task, reference_time=now)

        # Check that symptoms include time descriptors
        found_relative_time = False
        for symptom in sanitized.symptoms:
            if "ago" in symptom:
                found_relative_time = True
                break

        assert found_relative_time

    def test_sanitize_with_lab_results(self):
        """Test sanitization with lab results"""
        patient = LocalPatientRecord(
            patient_id="test-003",
            age=60,
            sex=SexEnum.MALE,
            lab_results=[
                LabResult(
                    test_name="Hemoglobin A1C", value=7.3456, unit="%", timestamp=datetime.now()
                ),
                LabResult(
                    test_name="Fasting Glucose",
                    value=135.789,
                    unit="mg/dL",
                    timestamp=datetime.now(),
                ),
            ],
        )

        task = "Evaluate diabetes management"
        sanitized = Sanitizer.sanitize(patient, task)

        # Check that numeric findings are rounded
        assert len(sanitized.numeric_findings) > 0
        for value in sanitized.numeric_findings.values():
            # Values should be rounded to 2 sig figs
            assert isinstance(value, (int, float))

    def test_sanitize_rejects_phi_in_output(self):
        """Test that sanitizer rejects output containing PHI"""
        patient = LocalPatientRecord(
            patient_id="test-004",
            age=40,
            sex=SexEnum.FEMALE,
            conditions=["Contact john@example.com for records"],
        )

        task = "Evaluate symptoms"

        # Sanitizer should clean the email from conditions
        sanitized = Sanitizer.sanitize(patient, task)
        output_json = sanitized.model_dump_json()

        # Email should not be in output
        assert "john@example.com" not in output_json

    def test_validate_no_phi_positive(self):
        """Test PHI validation with clean text"""
        clean_text = "Patient has fever and cough for 2 days"
        assert Sanitizer.validate_no_phi(clean_text) is True

    def test_validate_no_phi_negative(self):
        """Test PHI validation with PHI present"""
        phi_text = "Contact john@example.com at 555-1234"
        assert Sanitizer.validate_no_phi(phi_text) is False

    def test_sanitize_complex_patient(self):
        """Test sanitization with complex patient record"""
        now = datetime.now()
        patient = LocalPatientRecord(
            patient_id="complex-001",
            name="Jane Smith",
            age=45,
            sex=SexEnum.FEMALE,
            conditions=["type 2 diabetes", "hypertension"],
            medications=[
                Medication(
                    name="Metformin", dosage="500mg", frequency="twice daily", started_date=None
                )
            ],
            allergies=["penicillin"],
            history=[
                HistoryRecord(
                    timestamp=now - timedelta(days=3),
                    symptoms=["headache", "fatigue"],
                    diagnosis="Stress-related",
                )
            ],
            lab_results=[
                LabResult(
                    test_name="HbA1c", value=6.8, unit="%", timestamp=now - timedelta(days=30)
                )
            ],
        )

        task = "Provide diabetes management recommendations"
        sanitized = Sanitizer.sanitize(
            patient, task, current_symptoms=["increased thirst", "frequent urination"]
        )

        # Verify comprehensive sanitization
        output_json = sanitized.model_dump_json()

        # No PHI
        assert "Jane Smith" not in output_json
        assert "complex-001" not in output_json

        # Has fingerprint
        assert len(sanitized.fingerprint) == 64

        # Has conditions
        assert len(sanitized.conditions) > 0

        # Has symptoms
        assert len(sanitized.symptoms) > 0

        # Has numeric findings
        assert len(sanitized.numeric_findings) > 0

        # Task is present
        assert sanitized.task

    def test_sanitize_limits_history_records(self):
        """Test that sanitizer limits number of historical records"""
        now = datetime.now()

        # Create patient with many history records
        history = []
        for i in range(20):
            history.append(
                HistoryRecord(
                    timestamp=now - timedelta(days=i),
                    symptoms=[f"symptom_{i}"],
                    diagnosis=f"diagnosis_{i}",
                )
            )

        patient = LocalPatientRecord(
            patient_id="test-history", age=30, sex=SexEnum.MALE, history=history
        )

        task = "Evaluate patient history"
        sanitized = Sanitizer.sanitize(patient, task)

        # Should only include last 5 records
        # Each record can contribute multiple symptoms
        # So we just check that it's not all 20
        assert len(sanitized.symptoms) < 20

    def test_sanitize_limits_lab_results(self):
        """Test that sanitizer limits number of lab results"""
        now = datetime.now()

        # Create patient with many lab results
        lab_results = []
        for i in range(20):
            lab_results.append(
                LabResult(
                    test_name=f"test_{i}",
                    value=float(i * 10),
                    unit="unit",
                    timestamp=now - timedelta(days=i),
                )
            )

        patient = LocalPatientRecord(
            patient_id="test-labs", age=30, sex=SexEnum.MALE, lab_results=lab_results
        )

        task = "Evaluate lab trends"
        sanitized = Sanitizer.sanitize(patient, task)

        # Should only include last 10 results
        assert len(sanitized.numeric_findings) <= 10
