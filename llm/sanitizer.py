"""
Sanitizer module: De-identifies and sanitizes patient data before cloud transmission.

This module is CRITICAL for privacy. It ensures no raw PHI leaves local nodes.
All cloud-bound data must pass through this sanitizer.
"""

import hashlib
import re
from datetime import datetime

from llm.schemas import LocalPatientRecord, SanitizedPrompt


class Sanitizer:
    """
    Sanitizer class that removes PHI and creates sanitized prompts.

    Rules:
    - Remove all names, identifiers, exact dates
    - Convert absolute timestamps to relative descriptors (e.g., "2 days ago")
    - Round numeric lab values to 2 significant figures
    - Generate fingerprint for caching and auditing
    - Reject if sanitization cannot be done safely
    """

    # Patterns to detect and remove PHI
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b|\b\d{3}[-]\d{4}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    DATE_PATTERN = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b")

    # Common name patterns (simple heuristic)
    NAME_INDICATORS = ["mr.", "mrs.", "ms.", "dr.", "prof."]

    @staticmethod
    def _round_to_sig_figs(value: float, sig_figs: int = 2) -> float:
        """Round numeric value to specified significant figures"""
        if value == 0:
            return 0.0
        from math import floor, log10

        return round(value, -int(floor(log10(abs(value)))) + (sig_figs - 1))

    @staticmethod
    def _relative_time_descriptor(timestamp: datetime, reference: datetime | None = None) -> str:
        """
        Convert absolute timestamp to relative descriptor.
        Examples: "2 days ago", "1 week ago", "3 months ago"
        """
        if reference is None:
            reference = datetime.now()

        delta = reference - timestamp

        if delta.days < 1:
            hours = delta.seconds // 3600
            if hours < 1:
                return "less than 1 hour ago"
            return f"{hours} hours ago"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 30:
            weeks = delta.days // 7
            return f"{weeks} weeks ago"
        elif delta.days < 365:
            months = delta.days // 30
            return f"{months} months ago"
        else:
            years = delta.days // 365
            return f"{years} years ago"

    @staticmethod
    def _detect_phi(text: str) -> list[str]:
        """
        Detect potential PHI in text.
        Returns list of PHI types found.
        """
        phi_found = []

        if Sanitizer.EMAIL_PATTERN.search(text):
            phi_found.append("email")
        if Sanitizer.PHONE_PATTERN.search(text):
            phi_found.append("phone")
        if Sanitizer.SSN_PATTERN.search(text):
            phi_found.append("ssn")
        if Sanitizer.DATE_PATTERN.search(text):
            phi_found.append("date")

        # Check for name indicators
        text_lower = text.lower()
        for indicator in Sanitizer.NAME_INDICATORS:
            if indicator in text_lower:
                phi_found.append("name_indicator")
                break

        return phi_found

    @staticmethod
    def _remove_phi_from_text(text: str) -> str:
        """Remove detected PHI from text"""
        # Remove emails
        text = Sanitizer.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        # Remove phone numbers
        text = Sanitizer.PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
        # Remove SSN
        text = Sanitizer.SSN_PATTERN.sub("[SSN_REDACTED]", text)
        # Remove dates
        text = Sanitizer.DATE_PATTERN.sub("[DATE_REDACTED]", text)

        return text

    @staticmethod
    def _generate_fingerprint(patient: LocalPatientRecord, task: str) -> str:
        """
        Generate deterministic SHA256 fingerprint for caching and auditing.
        Uses only sanitized fields.
        """
        data = f"{patient.age}|{patient.sex.value}|{sorted(patient.conditions)}|{task}"
        return hashlib.sha256(data.encode()).hexdigest()

    @classmethod
    def sanitize(
        cls,
        patient: LocalPatientRecord,
        task: str,
        current_symptoms: list[str] | None = None,
        reference_time: datetime | None = None,
    ) -> SanitizedPrompt:
        """
        Sanitize patient record for cloud transmission.

        Args:
            patient: Local patient record (contains PHI)
            task: Task instruction for LLM
            current_symptoms: Current symptoms (optional)
            reference_time: Reference time for relative descriptors (defaults to now)

        Returns:
            SanitizedPrompt with no PHI

        Raises:
            ValueError: If PHI detected in sanitized output or sanitization fails
        """
        if reference_time is None:
            reference_time = datetime.now()

        # Extract and sanitize conditions
        conditions = [cls._remove_phi_from_text(cond) for cond in patient.conditions]

        # Extract and sanitize symptoms with relative time
        symptoms = []
        if current_symptoms:
            symptoms = [cls._remove_phi_from_text(symptom) for symptom in current_symptoms]

        # Add historical symptoms with relative time descriptors
        for record in patient.history[-5:]:  # Only last 5 records
            time_desc = cls._relative_time_descriptor(record.timestamp, reference_time)
            for symptom in record.symptoms:
                sanitized_symptom = cls._remove_phi_from_text(symptom)
                symptoms.append(f"{sanitized_symptom} ({time_desc})")

        # Extract and round lab values
        numeric_findings = {}
        for lab in patient.lab_results[-10:]:  # Only last 10 lab results
            key = cls._remove_phi_from_text(lab.test_name.lower().replace(" ", "_"))
            rounded_value = cls._round_to_sig_figs(lab.value, 2)
            numeric_findings[key] = rounded_value

        # Generate fingerprint
        fingerprint = cls._generate_fingerprint(patient, task)

        # Sanitize task instruction
        sanitized_task = cls._remove_phi_from_text(task)

        # Create sanitized prompt
        sanitized = SanitizedPrompt(
            fingerprint=fingerprint,
            age=patient.age,
            sex=patient.sex,
            conditions=conditions,
            symptoms=symptoms,
            numeric_findings=numeric_findings,
            task=sanitized_task,
        )

        # CRITICAL: Verify no PHI in output
        output_json = sanitized.model_dump_json()
        phi_detected = cls._detect_phi(output_json)

        if phi_detected:
            raise ValueError(
                f"SANITIZER_FAIL: PHI detected in sanitized output: {phi_detected}. "
                "Cannot send to cloud."
            )

        return sanitized

    @classmethod
    def validate_no_phi(cls, text: str) -> bool:
        """
        Validate that text contains no PHI.
        Returns True if clean, False if PHI detected.
        """
        phi_detected = cls._detect_phi(text)
        return len(phi_detected) == 0
