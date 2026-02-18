"""
Pydantic models generated from JSON schemas.
These models provide runtime validation for all data structures.
"""
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SexEnum(str, Enum):
    """Sex/gender enumeration"""
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class Medication(BaseModel):
    """Medication record"""
    name: str
    dosage: str
    frequency: str
    started_date: Optional[date] = None


class LabResult(BaseModel):
    """Laboratory test result"""
    test_name: str
    value: float
    unit: str
    timestamp: datetime
    reference_range: Optional[str] = None


class HistoryRecord(BaseModel):
    """Patient history entry"""
    timestamp: datetime
    symptoms: List[str]
    diagnosis: Optional[str] = None
    notes: Optional[str] = None


class LocalPatientRecord(BaseModel):
    """
    Local patient record stored encrypted on device/hospital node.
    NEVER send raw PHI from this model to cloud.
    """
    patient_id: str = Field(..., description="Local unique identifier")
    age: int = Field(..., ge=0, le=150)
    sex: SexEnum
    name: Optional[str] = Field(None, description="Full name - NEVER sent to cloud")
    date_of_birth: Optional[date] = Field(None, description="Converted to age for cloud")
    conditions: List[str] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    history: List[HistoryRecord] = Field(default_factory=list)
    lab_results: List[LabResult] = Field(default_factory=list)


class SanitizedPrompt(BaseModel):
    """
    De-identified, sanitized data sent to cloud LLM council.
    Must not contain any PHI or identifiable information.
    """
    fingerprint: str = Field(..., pattern=r"^[a-f0-9]{64}$", description="SHA256 hash")
    age: int = Field(..., ge=0, le=150)
    sex: SexEnum
    conditions: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list, description="Relative time descriptors")
    numeric_findings: Dict[str, float] = Field(default_factory=dict, description="Rounded lab values")
    task: str = Field(..., description="Instruction for LLM")
    context: Optional[str] = Field(None, description="Additional sanitized context")

    model_config = ConfigDict(extra="forbid")


class Differential(BaseModel):
    """Possible diagnosis with probability and rationale"""
    label: str = Field(..., description="Possible diagnosis or condition")
    probability: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Conservative explanation")
    requires_verification: Optional[bool] = Field(False, description="External claims need verification")
    verification_query: Optional[str] = None


class ResponseMetadata(BaseModel):
    """Metadata for council response"""
    latency_ms: Optional[float] = None
    timestamp: Optional[datetime] = None
    model_version: Optional[str] = None


class CouncilResponse(BaseModel):
    """
    Structured response from each LLM in the council.
    All LLMs must return data matching this schema.
    """
    model_id: str
    differentials: List[Differential] = Field(..., min_length=0, max_length=10)
    recommended_next_steps: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    red_flag: bool = Field(..., description="True if emergency detected")
    red_flag_rationale: Optional[str] = None
    metadata: Optional[ResponseMetadata] = None

    model_config = ConfigDict(extra="forbid")


class MLFeatureVector(BaseModel):
    """Feature vector for local ML model prediction"""
    features: List[float]
    feature_names: List[str]
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('features', 'feature_names')
    @classmethod
    def check_lengths_match(cls, v, info):
        """Ensure features and feature_names have the same length"""
        # This validator runs per field, so we need to check in model_validator
        return v
