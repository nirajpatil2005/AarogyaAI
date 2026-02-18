#!/usr/bin/env python3
"""
Demo script showing AarogyaAI Phase 0 capabilities.

This script demonstrates:
1. Sanitizer removing PHI
2. Red-flag engine detecting emergencies
3. Complete query flow with privacy protection
"""

from datetime import datetime, timedelta

from llm.red_flags import RedFlagEngine
from llm.sanitizer import Sanitizer
from llm.schemas import (
    HistoryRecord,
    LabResult,
    LocalPatientRecord,
    SexEnum,
)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_sanitizer():
    """Demo: Sanitizer removes PHI and creates safe prompts"""
    print_section("DEMO 1: Sanitizer - Privacy Protection")

    # Create a patient record with PHI (simulated)
    patient = LocalPatientRecord(
        patient_id="P12345",
        name="John Doe",  # PHI - will NOT be sent to cloud
        age=45,
        sex=SexEnum.MALE,
        conditions=["hypertension", "type 2 diabetes"],
        history=[
            HistoryRecord(
                timestamp=datetime.now() - timedelta(days=3),
                symptoms=["headache", "fatigue"],
                diagnosis="Stress-related",
            )
        ],
        lab_results=[
            LabResult(
                test_name="HbA1c",
                value=7.2,
                unit="%",
                timestamp=datetime.now() - timedelta(days=30),
            ),
            LabResult(
                test_name="Blood Pressure",
                value=142.5,
                unit="mmHg",
                timestamp=datetime.now() - timedelta(days=7),
            ),
        ],
    )

    print("Original Patient Record (CONTAINS PHI - NEVER SENT TO CLOUD):")
    print(f"  Name: {patient.name}")
    print(f"  Patient ID: {patient.patient_id}")
    print(f"  Age: {patient.age}")
    print(f"  Conditions: {patient.conditions}")
    print(f"  History entries: {len(patient.history)}")
    print(f"  Lab results: {len(patient.lab_results)}")

    # Sanitize the record
    task = (
        "Return JSON only: {differentials, recommended_next_steps, confidence, red_flag}. "
        "Use conservative clinical language."
    )

    sanitized = Sanitizer.sanitize(
        patient=patient, task=task, current_symptoms=["increased thirst", "frequent urination"]
    )

    print("\n✅ Sanitized Prompt (SAFE FOR CLOUD):")
    print(f"  Fingerprint: {sanitized.fingerprint[:16]}...")
    print(f"  Age: {sanitized.age}")
    print(f"  Sex: {sanitized.sex}")
    print(f"  Conditions: {sanitized.conditions}")
    print(f"  Symptoms: {sanitized.symptoms}")
    print(f"  Numeric findings: {sanitized.numeric_findings}")

    # Verify no PHI
    output = sanitized.model_dump_json()
    is_clean = Sanitizer.validate_no_phi(output)
    print(f"\n✅ PHI Validation: {'CLEAN - No PHI detected' if is_clean else 'FAILED'}")

    print("\n🔒 Privacy Guarantee:")
    print("  - Patient name removed")
    print("  - Patient ID removed")
    print("  - Dates converted to relative time")
    print("  - Lab values rounded")
    print("  - Fingerprint for auditing")


def demo_red_flags():
    """Demo: Red-flag engine detecting emergencies"""
    print_section("DEMO 2: Red-Flag Engine - Emergency Detection")

    # Test 1: Routine symptoms
    print("Test 1: Routine Symptoms")
    symptoms = ["mild headache", "runny nose", "slight fatigue"]
    result = RedFlagEngine.evaluate(symptoms)
    print(f"  Symptoms: {symptoms}")
    print(f"  Emergency: {result.is_emergency}")
    print(f"  Urgency: {result.urgency_level}")
    print(f"  Rationale: {result.rationale}")

    # Test 2: Emergency symptoms
    print("\n\nTest 2: EMERGENCY Symptoms")
    symptoms = ["severe chest pain", "shortness of breath", "sweating"]
    result = RedFlagEngine.evaluate(symptoms)
    print(f"  Symptoms: {symptoms}")
    print(f"  Emergency: {result.is_emergency} ⚠️")
    print(f"  Urgency: {result.urgency_level.upper()}")
    print(f"  Triggered rules: {result.triggered_rules}")
    print(f"  Rationale: {result.rationale}")

    # Test 3: With vital signs
    print("\n\nTest 3: Urgent Symptoms with Abnormal Vitals")
    symptoms = ["fever", "rapid breathing"]
    vitals = {"temperature_f": 103.5, "respiratory_rate": 28, "oxygen_saturation": 91.0}
    result = RedFlagEngine.evaluate(symptoms, vitals)
    print(f"  Symptoms: {symptoms}")
    print(f"  Vitals: {vitals}")
    print(f"  Emergency: {result.is_emergency}")
    print(f"  Urgency: {result.urgency_level}")
    print(f"  Triggered rules: {result.triggered_rules}")


def demo_complete_flow():
    """Demo: Complete query flow"""
    print_section("DEMO 3: Complete Query Flow")

    # Simulate a patient query
    patient = LocalPatientRecord(
        patient_id="P67890",
        name="Jane Smith",
        age=38,
        sex=SexEnum.FEMALE,
        conditions=["asthma"],
    )

    symptoms = ["shortness of breath", "wheezing", "chest tightness"]

    print("Patient Query:")
    print(f"  Symptoms: {symptoms}")

    # Step 1: Check red flags FIRST (local, immediate)
    print("\n1️⃣ Red-Flag Check (LOCAL):")
    red_flag_result = RedFlagEngine.evaluate(symptoms)
    print(f"  Emergency: {red_flag_result.is_emergency}")
    print(f"  Urgency: {red_flag_result.urgency_level}")

    if red_flag_result.is_emergency:
        print("\n⚠️  EMERGENCY DETECTED - Immediate action required")
        print("  Skipping cloud consultation")
        print("  Providing emergency guidance to user")
        return

    # Step 2: Sanitize for cloud (if not emergency)
    print("\n2️⃣ Sanitization (PRIVACY PROTECTION):")
    task = "Evaluate respiratory symptoms and provide recommendations"
    sanitized = Sanitizer.sanitize(patient, task, symptoms)
    print("  Sanitized: ✅")
    print(f"  Fingerprint: {sanitized.fingerprint[:16]}...")

    # Step 3: Would send to cloud (Phase 2)
    print("\n3️⃣ Cloud Consultation (Phase 2 - Not Yet Implemented):")
    print("  Would send sanitized prompt to LLM council")
    print("  No PHI transmitted")
    print("  Response aggregated and validated")

    # Step 4: Return recommendations
    print("\n4️⃣ Recommendations:")
    print("  - Contact healthcare provider")
    print("  - Use prescribed rescue inhaler if available")
    print("  - Monitor symptoms closely")
    print("  - Seek urgent care if symptoms worsen")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print(" " * 20 + "AarogyaAI - Phase 0 Demo")
    print(" " * 15 + "Privacy-First Medical Intelligence")
    print("=" * 80)

    try:
        demo_sanitizer()
        input("\n\nPress Enter to continue to next demo...")

        demo_red_flags()
        input("\n\nPress Enter to continue to next demo...")

        demo_complete_flow()

        print_section("DEMO COMPLETE ✅")
        print("Phase 0 demonstrates:")
        print("  ✅ Privacy protection via sanitization")
        print("  ✅ Emergency detection with red-flag engine")
        print("  ✅ Local-first processing")
        print("  ✅ No PHI leaves device")
        print("\nNext: Phase 1 will add local ML models and encrypted storage")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError in demo: {e}")
        raise


if __name__ == "__main__":
    main()
