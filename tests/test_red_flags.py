"""
Unit tests for red-flag rule engine.
Testing emergency detection, positive and negative cases.
"""
import pytest
from llm.red_flags import RedFlagEngine, RedFlagResult


class TestRedFlagEngine:
    """Test suite for RedFlagEngine"""
    
    def test_immediate_chest_pain(self):
        """Test immediate red flag for severe chest pain"""
        symptoms = ["severe chest pain", "sweating"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
        assert "severe chest pain" in result.triggered_rules
        assert "IMMEDIATE EMERGENCY" in result.rationale
    
    def test_immediate_hemoptysis(self):
        """Test immediate red flag for coughing up blood"""
        symptoms = ["coughing up blood", "shortness of breath"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
        assert len(result.triggered_rules) > 0
    
    def test_immediate_syncope(self):
        """Test immediate red flag for loss of consciousness"""
        symptoms = ["syncope", "weakness"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
        assert "syncope" in result.triggered_rules
    
    def test_immediate_stroke_symptoms(self):
        """Test immediate red flag for stroke symptoms"""
        symptoms = ["sudden weakness", "facial drooping", "slurred speech"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
    
    def test_urgent_chest_pain(self):
        """Test urgent flag for non-severe chest pain"""
        symptoms = ["chest pain", "mild discomfort"]
        result = RedFlagEngine.evaluate(symptoms)
        
        # Should be urgent but not immediate emergency
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
        assert "URGENT" in result.rationale
    
    def test_urgent_persistent_fever(self):
        """Test urgent flag for persistent fever"""
        symptoms = ["persistent fever", "body aches"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
    
    def test_combination_cardiac_risk(self):
        """Test combination rule for cardiac risk"""
        symptoms = ["chest pain", "shortness of breath", "sweating"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
        assert "cardiac" in result.rationale.lower() or "cardiac_risk" in result.triggered_rules
    
    def test_combination_respiratory_distress(self):
        """Test combination rule for respiratory distress"""
        symptoms = ["shortness of breath", "chest pain", "rapid breathing"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
    
    def test_no_red_flags_routine(self):
        """Test routine symptoms with no red flags"""
        symptoms = ["mild headache", "runny nose"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is False
        assert result.urgency_level == "routine"
        assert len(result.triggered_rules) == 0
        assert "No immediate red flags" in result.rationale
    
    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive"""
        symptoms = ["SEVERE CHEST PAIN", "Sweating"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
    
    def test_vital_signs_low_oxygen(self):
        """Test vital sign threshold for low oxygen saturation"""
        symptoms = ["shortness of breath"]
        vitals = {"oxygen_saturation": 88.0}
        
        result = RedFlagEngine.evaluate(symptoms, vitals)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
        assert "oxygen_saturation" in result.rationale
    
    def test_vital_signs_high_heart_rate(self):
        """Test vital sign threshold for high heart rate"""
        symptoms = ["palpitations"]
        vitals = {"heart_rate": 140}
        
        result = RedFlagEngine.evaluate(symptoms, vitals)
        
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
    
    def test_vital_signs_low_blood_pressure(self):
        """Test vital sign threshold for low blood pressure"""
        symptoms = ["dizziness"]
        vitals = {"systolic_bp": 80, "diastolic_bp": 50}
        
        result = RedFlagEngine.evaluate(symptoms, vitals)
        
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
    
    def test_vital_signs_high_temperature(self):
        """Test vital sign threshold for high fever"""
        symptoms = ["fever"]
        vitals = {"temperature_f": 104.0}
        
        result = RedFlagEngine.evaluate(symptoms, vitals)
        
        assert result.urgency_level == "urgent"
        assert "temperature_f" in result.rationale
    
    def test_vital_signs_normal(self):
        """Test that normal vital signs don't trigger flags"""
        symptoms = ["mild cough"]
        vitals = {
            "heart_rate": 75,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "respiratory_rate": 16,
            "temperature_f": 98.6,
            "oxygen_saturation": 98.0
        }
        
        result = RedFlagEngine.evaluate(symptoms, vitals)
        
        # Mild cough should remain routine
        assert result.urgency_level == "routine"
        assert result.is_emergency is False
    
    def test_multiple_immediate_flags(self):
        """Test multiple immediate red flags"""
        symptoms = [
            "severe chest pain",
            "crushing chest pain",
            "shortness of breath",
            "syncope"
        ]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
        assert len(result.triggered_rules) >= 2
    
    def test_suicidal_thoughts_immediate(self):
        """Test that suicidal thoughts trigger immediate flag"""
        symptoms = ["suicidal thoughts", "depression"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
    
    def test_anaphylaxis_immediate(self):
        """Test anaphylaxis triggers immediate flag"""
        symptoms = ["throat swelling", "difficulty breathing", "severe allergic reaction"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
    
    def test_seizure_immediate(self):
        """Test seizure triggers immediate flag"""
        symptoms = ["seizure", "convulsion"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is True
        assert result.urgency_level == "immediate"
    
    def test_blood_in_stool_urgent(self):
        """Test blood in stool is flagged as urgent"""
        symptoms = ["blood in stool", "abdominal pain"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
    
    def test_sudden_vision_loss_urgent(self):
        """Test sudden vision loss is urgent"""
        symptoms = ["sudden vision loss", "eye pain"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
    
    def test_confusion_urgent(self):
        """Test confusion is flagged as urgent"""
        symptoms = ["confusion", "altered mental status"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.urgency_level == "urgent"
        assert len(result.triggered_rules) > 0
    
    def test_partial_symptom_matching(self):
        """Test that partial matches work correctly"""
        # "chest pain" should match "chest pain radiating to arm"
        symptoms = ["I have chest pain that moves to my left arm"]
        result = RedFlagEngine.evaluate(symptoms)
        
        # Should trigger cardiac-related flag
        assert result.is_emergency is True or result.urgency_level == "urgent"
    
    def test_empty_symptoms_list(self):
        """Test handling of empty symptoms list"""
        symptoms = []
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is False
        assert result.urgency_level == "routine"
        assert len(result.triggered_rules) == 0
    
    def test_normalize_text(self):
        """Test text normalization helper"""
        assert RedFlagEngine._normalize_text("  CHEST PAIN  ") == "chest pain"
        assert RedFlagEngine._normalize_text("Fever") == "fever"
    
    def test_rationale_contains_action_items(self):
        """Test that rationale includes actionable guidance"""
        symptoms = ["severe chest pain"]
        result = RedFlagEngine.evaluate(symptoms)
        
        # Should include guidance about seeking care
        assert "emergency" in result.rationale.lower() or "immediate" in result.rationale.lower()
        
    def test_combination_sepsis_risk(self):
        """Test sepsis combination rule"""
        symptoms = ["fever", "confusion", "rapid heart rate"]
        result = RedFlagEngine.evaluate(symptoms)
        
        # Should trigger sepsis combination rule or individual urgent flags
        assert result.urgency_level in ["immediate", "urgent"]
    
    def test_negative_case_common_cold(self):
        """Test common cold symptoms don't trigger red flags"""
        symptoms = ["runny nose", "sneezing", "mild sore throat", "congestion"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is False
        assert result.urgency_level == "routine"
        assert len(result.triggered_rules) == 0
    
    def test_negative_case_mild_symptoms(self):
        """Test mild symptoms don't trigger red flags"""
        symptoms = ["mild headache", "slight fatigue", "occasional cough"]
        result = RedFlagEngine.evaluate(symptoms)
        
        assert result.is_emergency is False
        assert result.urgency_level == "routine"
    
    def test_edge_case_vital_thresholds(self):
        """Test edge cases at vital sign thresholds"""
        # Exactly at threshold should be OK
        symptoms = ["routine checkup"]
        vitals = {"oxygen_saturation": 92.0}  # Exactly at minimum
        
        result = RedFlagEngine.evaluate(symptoms, vitals)
        
        # At threshold should be OK (not below)
        assert result.urgency_level == "routine"
