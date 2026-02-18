"""
Red-flag rule engine: Detects potential emergencies that require immediate action.

This module implements conservative, evidence-based rules for emergency detection.
If any red flag is triggered, the system must immediately escalate to emergency protocols.
"""

from dataclasses import dataclass


@dataclass
class RedFlagResult:
    """Result from red-flag evaluation"""

    is_emergency: bool
    rationale: str
    triggered_rules: list[str]
    urgency_level: str  # "immediate", "urgent", "routine"


class RedFlagEngine:
    """
    Red-flag rule engine for emergency detection.

    Rules are conservative and based on clinical emergency indicators.
    If any red flag triggers, system should:
    1. Immediately notify user
    2. Provide emergency contact information
    3. Skip normal AI processing
    """

    # Critical symptoms that require immediate evaluation
    IMMEDIATE_RED_FLAGS = [
        "severe chest pain",
        "crushing chest pain",
        "chest pain radiating to arm",
        "chest pain radiating to jaw",
        "sudden severe headache",
        "worst headache of life",
        "syncope",
        "loss of consciousness",
        "uncontrolled bleeding",
        "severe bleeding",
        "hemoptysis",
        "coughing up blood",
        "severe shortness of breath",
        "difficulty breathing",
        "unable to breathe",
        "stroke symptoms",
        "facial drooping",
        "slurred speech",
        "sudden weakness",
        "severe allergic reaction",
        "anaphylaxis",
        "throat swelling",
        "severe abdominal pain",
        "rigid abdomen",
        "suicidal thoughts",
        "suicide",
        "self harm",
        "seizure",
        "convulsion",
    ]

    # Urgent symptoms that need prompt evaluation (within hours)
    URGENT_FLAGS = [
        "chest pain",
        "chest discomfort",
        "shortness of breath",
        "difficulty breathing on exertion",
        "persistent fever",
        "high fever",
        "severe pain",
        "sudden vision loss",
        "sudden hearing loss",
        "severe headache",
        "persistent vomiting",
        "severe diarrhea",
        "blood in stool",
        "blood in urine",
        "severe dizziness",
        "confusion",
        "altered mental status",
    ]

    # Combination rules (multiple symptoms together)
    COMBINATION_RULES = [
        {
            "name": "cardiac_risk",
            "symptoms": ["chest pain", "shortness of breath", "sweating"],
            "threshold": 2,
            "level": "immediate",
            "rationale": "Multiple cardiac symptoms present",
        },
        {
            "name": "sepsis_risk",
            "symptoms": ["fever", "confusion", "rapid heart rate", "low blood pressure"],
            "threshold": 2,
            "level": "immediate",
            "rationale": "Possible sepsis - requires immediate evaluation",
        },
        {
            "name": "respiratory_distress",
            "symptoms": ["shortness of breath", "chest pain", "rapid breathing"],
            "threshold": 2,
            "level": "immediate",
            "rationale": "Respiratory distress pattern",
        },
    ]

    # Vital sign thresholds (if available)
    VITAL_THRESHOLDS = {
        "heart_rate": {"min": 40, "max": 120, "level": "urgent"},
        "systolic_bp": {"min": 90, "max": 180, "level": "urgent"},
        "diastolic_bp": {"min": 60, "max": 110, "level": "urgent"},
        "respiratory_rate": {"min": 10, "max": 25, "level": "urgent"},
        "temperature_f": {"min": 95.0, "max": 103.0, "level": "urgent"},
        "oxygen_saturation": {"min": 92.0, "max": 100.0, "level": "immediate"},
    }

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize text for matching"""
        return text.lower().strip()

    @classmethod
    def _check_immediate_flags(cls, symptoms: list[str]) -> tuple[bool, list[str], str]:
        """Check for immediate emergency flags"""
        triggered = []
        normalized_symptoms = [cls._normalize_text(s) for s in symptoms]

        for flag in cls.IMMEDIATE_RED_FLAGS:
            for symptom in normalized_symptoms:
                if flag in symptom:
                    triggered.append(flag)

        if triggered:
            rationale = (
                f"IMMEDIATE EMERGENCY: Detected critical symptoms: {', '.join(triggered)}. "
                "Seek emergency care immediately or call emergency services."
            )
            return True, triggered, rationale

        return False, [], ""

    @classmethod
    def _check_urgent_flags(cls, symptoms: list[str]) -> tuple[bool, list[str], str]:
        """Check for urgent flags requiring prompt evaluation"""
        triggered = []
        normalized_symptoms = [cls._normalize_text(s) for s in symptoms]

        for flag in cls.URGENT_FLAGS:
            for symptom in normalized_symptoms:
                if flag in symptom:
                    triggered.append(flag)

        if triggered:
            rationale = (
                f"URGENT: Detected symptoms requiring prompt evaluation: {', '.join(triggered)}. "
                "Contact your healthcare provider today or visit urgent care."
            )
            return True, triggered, rationale

        return False, [], ""

    @classmethod
    def _check_combination_rules(cls, symptoms: list[str]) -> tuple[bool, list[str], str]:
        """Check for dangerous symptom combinations"""
        normalized_symptoms = [cls._normalize_text(s) for s in symptoms]

        for rule in cls.COMBINATION_RULES:
            matches = 0
            matched_symptoms = []

            for rule_symptom in rule["symptoms"]:
                for user_symptom in normalized_symptoms:
                    if rule_symptom in user_symptom:
                        matches += 1
                        matched_symptoms.append(rule_symptom)
                        break

            if matches >= rule["threshold"]:
                rationale = (
                    f"{rule['level'].upper()}: {rule['rationale']}. "
                    f"Matched symptoms: {', '.join(matched_symptoms)}. "
                    "Seek immediate medical attention."
                )
                return True, [rule["name"]], rationale

        return False, [], ""

    @classmethod
    def _check_vital_signs(cls, vitals: dict) -> tuple[bool, list[str], str]:
        """Check vital signs against thresholds"""
        triggered = []
        is_emergency = False

        for vital_name, vital_value in vitals.items():
            if vital_name in cls.VITAL_THRESHOLDS:
                threshold = cls.VITAL_THRESHOLDS[vital_name]
                if vital_value < threshold["min"] or vital_value > threshold["max"]:
                    triggered.append(f"{vital_name}={vital_value}")
                    if threshold["level"] == "immediate":
                        is_emergency = True

        if triggered:
            level = "IMMEDIATE EMERGENCY" if is_emergency else "URGENT"
            rationale = (
                f"{level}: Vital signs outside safe range: {', '.join(triggered)}. "
                "Seek medical attention."
            )
            return True, triggered, rationale

        return False, [], ""

    @classmethod
    def evaluate(cls, symptoms: list[str], vitals: dict | None = None) -> RedFlagResult:
        """
        Evaluate symptoms and vitals for red flags.

        Args:
            symptoms: List of symptom descriptions
            vitals: Optional dict of vital signs

        Returns:
            RedFlagResult with emergency status and rationale
        """
        all_triggered = []
        urgency_level = "routine"
        rationale = "No immediate red flags detected."

        # Check immediate emergency flags first
        is_immediate, triggered, msg = cls._check_immediate_flags(symptoms)
        if is_immediate:
            all_triggered.extend(triggered)
            urgency_level = "immediate"
            rationale = msg
            return RedFlagResult(
                is_emergency=True,
                rationale=rationale,
                triggered_rules=all_triggered,
                urgency_level=urgency_level,
            )

        # Check combination rules
        is_combo, triggered, msg = cls._check_combination_rules(symptoms)
        if is_combo:
            all_triggered.extend(triggered)
            urgency_level = "immediate"
            rationale = msg
            return RedFlagResult(
                is_emergency=True,
                rationale=rationale,
                triggered_rules=all_triggered,
                urgency_level=urgency_level,
            )

        # Check urgent flags
        is_urgent, triggered, msg = cls._check_urgent_flags(symptoms)
        if is_urgent:
            all_triggered.extend(triggered)
            urgency_level = "urgent"
            rationale = msg

        # Check vital signs if provided
        if vitals:
            is_vital_issue, triggered, msg = cls._check_vital_signs(vitals)
            if is_vital_issue:
                all_triggered.extend(triggered)
                if "IMMEDIATE" in msg:
                    urgency_level = "immediate"
                    rationale = msg
                    return RedFlagResult(
                        is_emergency=True,
                        rationale=rationale,
                        triggered_rules=all_triggered,
                        urgency_level=urgency_level,
                    )
                elif urgency_level != "immediate":
                    urgency_level = "urgent"
                    rationale = msg

        # Determine if this is an emergency
        is_emergency = urgency_level == "immediate"

        return RedFlagResult(
            is_emergency=is_emergency,
            rationale=rationale,
            triggered_rules=all_triggered,
            urgency_level=urgency_level,
        )
