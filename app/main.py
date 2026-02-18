"""
Main FastAPI application for AarogyaAI local node.
Provides REST API for local health queries with privacy-first design.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from llm.schemas import LocalPatientRecord, SexEnum
from llm.sanitizer import Sanitizer
from llm.red_flags import RedFlagEngine

# Create FastAPI app
app = FastAPI(
    title="AarogyaAI Local Node",
    description="Privacy-first medical intelligence - local node API",
    version="0.1.0"
)


class QueryRequest(BaseModel):
    """Request model for health query"""
    device_id: str = Field(..., description="Local device identifier")
    patient_id: Optional[str] = Field(None, description="Patient identifier (local only)")
    symptoms: List[str] = Field(..., description="Current symptoms")
    vitals: Optional[Dict[str, float]] = Field(None, description="Optional vital signs")


class QueryResponse(BaseModel):
    """Response model for health query"""
    status: str
    urgency_level: str
    red_flag: bool
    message: str
    recommendations: List[str]
    fingerprint: Optional[str] = None
    timestamp: datetime


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "service": "AarogyaAI Local Node",
        "version": "0.1.0",
        "status": "running",
        "privacy": "local-first, no PHI upload by default"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/query", response_model=QueryResponse)
async def query_symptoms(request: QueryRequest):
    """
    Query endpoint for symptom evaluation.
    
    Flow:
    1. Check red flags first (emergency detection)
    2. If emergency, return immediate guidance
    3. Otherwise, sanitize and prepare for ML/LLM processing
    4. Return conservative recommendations
    
    Note: This MVP version does local processing only.
    Council integration will be added in Phase 2.
    """
    try:
        # Step 1: Evaluate red flags FIRST
        red_flag_result = RedFlagEngine.evaluate(
            symptoms=request.symptoms,
            vitals=request.vitals
        )
        
        # Step 2: If emergency detected, return immediately
        if red_flag_result.is_emergency:
            return QueryResponse(
                status="EMERGENCY",
                urgency_level=red_flag_result.urgency_level,
                red_flag=True,
                message=red_flag_result.rationale,
                recommendations=[
                    "SEEK IMMEDIATE MEDICAL ATTENTION",
                    "Call emergency services (911) or go to nearest emergency room",
                    "Do not delay - this may be a medical emergency"
                ],
                timestamp=datetime.now()
            )
        
        # Step 3: Create mock patient record for sanitization demo
        # In production, this would be retrieved from encrypted local storage
        mock_patient = LocalPatientRecord(
            patient_id=request.patient_id or request.device_id,
            age=35,  # Would come from stored record
            sex=SexEnum.UNKNOWN,  # Would come from stored record
            conditions=[]  # Would come from stored record
        )
        
        # Step 4: Sanitize the data (demo of privacy protection)
        task = (
            "Return JSON only: {differentials, recommended_next_steps, confidence, red_flag}. "
            "Use conservative clinical language. No identifiers."
        )
        
        try:
            sanitized = Sanitizer.sanitize(
                patient=mock_patient,
                task=task,
                current_symptoms=request.symptoms
            )
            fingerprint = sanitized.fingerprint
        except ValueError as e:
            # Sanitizer rejected the data
            return QueryResponse(
                status="ERROR",
                urgency_level="routine",
                red_flag=False,
                message=f"Data sanitization failed: {str(e)}",
                recommendations=["Please contact support"],
                timestamp=datetime.now()
            )
        
        # Step 5: Return conservative recommendations
        # In Phase 2, this would include council consultation
        if red_flag_result.urgency_level == "urgent":
            recommendations = [
                red_flag_result.rationale,
                "Contact your healthcare provider today",
                "Visit urgent care if symptoms worsen",
                "Monitor symptoms closely"
            ]
        else:
            recommendations = [
                "Monitor your symptoms",
                "Stay hydrated and rest",
                "Contact healthcare provider if symptoms persist or worsen",
                "This is not a substitute for professional medical advice"
            ]
        
        return QueryResponse(
            status="SUCCESS",
            urgency_level=red_flag_result.urgency_level,
            red_flag=False,
            message="Query processed successfully. Data sanitized and ready for analysis.",
            recommendations=recommendations,
            fingerprint=fingerprint,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        # Log error (in production, use structured logging)
        print(f"Error processing query: {str(e)}")
        
        # Return safe error response
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "PROCESSING_ERROR",
                "user_message": (
                    "Unable to process request. Please try again or contact your clinician."
                )
            }
        )


@app.post("/validate-sanitization")
async def validate_sanitization(text: str):
    """
    Utility endpoint to validate that text contains no PHI.
    Used for testing and validation.
    """
    is_clean = Sanitizer.validate_no_phi(text)
    
    if is_clean:
        return {
            "status": "CLEAN",
            "message": "No PHI detected in text"
        }
    else:
        phi_types = Sanitizer._detect_phi(text)
        return {
            "status": "PHI_DETECTED",
            "message": "PHI found in text",
            "phi_types": phi_types
        }


@app.get("/info")
async def get_info():
    """Get information about the service and its privacy guarantees"""
    return {
        "service": "AarogyaAI Local Node",
        "version": "0.1.0",
        "architecture": {
            "local_storage": "Encrypted SQLCipher database",
            "sanitizer": "Mandatory PHI removal before cloud transmission",
            "red_flag_engine": "Local emergency detection",
            "ml_model": "Local XGBoost for predictions",
            "council": "Cloud LLM council with sanitized prompts (Phase 2)",
            "federation": "Flower-based federated learning (Phase 3)"
        },
        "privacy_guarantees": [
            "No raw PHI leaves local node by default",
            "All cloud-bound data is sanitized and validated",
            "Local-first processing with optional cloud augmentation",
            "Red-flag detection runs locally",
            "Encrypted local storage",
            "Audit logging with fingerprints"
        ],
        "phases": {
            "phase_0": "Scaffold, schemas, sanitizer, red-flags (CURRENT)",
            "phase_1": "Local-only prediction and adjudication",
            "phase_2": "Council integration with structured aggregation",
            "phase_3": "Federated learning client and aggregator",
            "phase_4": "Hardening and clinical validation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
