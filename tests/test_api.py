"""
Unit tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPI:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AarogyaAI Local Node"
        assert data["status"] == "running"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_info_endpoint(self):
        """Test info endpoint returns architecture details"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "privacy_guarantees" in data
        assert "phases" in data
    
    def test_query_routine_symptoms(self):
        """Test query with routine symptoms"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-001",
                "symptoms": ["mild headache", "runny nose"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["urgency_level"] == "routine"
        assert data["red_flag"] is False
        assert len(data["recommendations"]) > 0
    
    def test_query_urgent_symptoms(self):
        """Test query with urgent symptoms"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-002",
                "symptoms": ["chest pain", "shortness of breath"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["urgency_level"] in ["urgent", "immediate"]
    
    def test_query_emergency_symptoms(self):
        """Test query with emergency symptoms"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-003",
                "symptoms": ["severe chest pain", "syncope"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "EMERGENCY"
        assert data["urgency_level"] == "immediate"
        assert data["red_flag"] is True
        assert "IMMEDIATE" in data["message"]
    
    def test_query_with_vitals(self):
        """Test query with vital signs"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-004",
                "symptoms": ["fever"],
                "vitals": {
                    "temperature_f": 102.0,
                    "heart_rate": 90,
                    "oxygen_saturation": 96.0
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
    
    def test_query_low_oxygen_emergency(self):
        """Test query with dangerously low oxygen triggers emergency"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-005",
                "symptoms": ["shortness of breath"],
                "vitals": {"oxygen_saturation": 85.0}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "EMERGENCY"
        assert data["red_flag"] is True
    
    def test_validate_sanitization_clean(self):
        """Test sanitization validation with clean text"""
        response = client.post(
            "/validate-sanitization",
            params={"text": "Patient has fever and cough"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "CLEAN"
    
    def test_validate_sanitization_phi_detected(self):
        """Test sanitization validation with PHI"""
        response = client.post(
            "/validate-sanitization",
            params={"text": "Contact john@example.com for records"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PHI_DETECTED"
        assert "email" in data["phi_types"]
    
    def test_query_returns_fingerprint(self):
        """Test that query returns fingerprint for auditing"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-006",
                "symptoms": ["cough"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # Non-emergency responses should include fingerprint
        if data["status"] == "SUCCESS":
            assert data["fingerprint"] is not None
            assert len(data["fingerprint"]) == 64  # SHA256 hex
    
    def test_query_missing_symptoms(self):
        """Test query with missing required fields"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-007"
                # Missing symptoms
            }
        )
        # Should return validation error
        assert response.status_code == 422
    
    def test_query_empty_symptoms(self):
        """Test query with empty symptoms list"""
        response = client.post(
            "/query",
            json={
                "device_id": "test-device-008",
                "symptoms": []
            }
        )
        # Should still process but return routine
        assert response.status_code == 200
        data = response.json()
        assert data["urgency_level"] == "routine"
