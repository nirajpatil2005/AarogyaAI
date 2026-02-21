"""
Red-Flag Engine — deterministic triage overrides.
Runs BEFORE the LLM council. If triggered, bypasses cloud calls entirely.
PII never leaves the device for emergency cases.
"""

RED_FLAG_SYMPTOMS = [
    "chest pain radiating",
    "can only speak in short sentences",
    "uncontrollable bleeding",
    "sudden weakness",
    "sudden numbness",
    "slurred speech",
    "overdose",
    "toxic ingestion",
    "loss of consciousness",
    "not breathing",
    "severe head injury",
]


def check_vital_signs(vitals: dict) -> tuple[bool, str]:
    """
    Check vital signs against hard-coded thresholds.
    Returns (is_emergency, reason).
    """
    hr = vitals.get("heart_rate")
    spo2 = vitals.get("spo2")
    rr = vitals.get("respiratory_rate")
    sbp = vitals.get("systolic_bp")
    temp_c = vitals.get("temperature_c")

    if hr is not None and (hr > 110 or hr < 50):
        return True, f"Critical heart rate: {hr} bpm"
    if spo2 is not None and spo2 < 94:
        return True, f"Critical SpO2: {spo2}%"
    if rr is not None and (rr > 24 or rr < 8):
        return True, f"Critical respiratory rate: {rr}/min"
    if sbp is not None and (sbp > 180 or sbp < 80):
        return True, f"Critical blood pressure: {sbp} mmHg systolic"
    if temp_c is not None and (temp_c > 39.5 or temp_c < 35.0):
        return True, f"Critical temperature: {temp_c}°C"

    return False, ""


def check_symptom_text(text: str) -> tuple[bool, str]:
    """
    Scan free-text symptom description for red-flag keywords.
    Returns (is_emergency, matched_phrase).
    """
    lower_text = text.lower()
    for phrase in RED_FLAG_SYMPTOMS:
        if phrase in lower_text:
            return True, phrase
    return False, ""


def evaluate(symptom_text: str, vitals: dict | None = None) -> dict:
    """
    Full red-flag evaluation. Returns a structured result.
    """
    vitals = vitals or {}

    vital_flag, vital_reason = check_vital_signs(vitals)
    symptom_flag, symptom_reason = check_symptom_text(symptom_text)

    is_emergency = vital_flag or symptom_flag
    reason = vital_reason or symptom_reason

    return {
        "is_emergency": is_emergency,
        "reason": reason,
        "action": "CALL_EMERGENCY_SERVICES" if is_emergency else "PROCEED_TO_COUNCIL",
        "message": (
            "⚠️ EMERGENCY DETECTED. Please call emergency services (112/911) immediately. "
            f"Reason: {reason}"
            if is_emergency
            else "No immediate emergency detected. Proceeding to council analysis."
        ),
    }
