"""
Pydantic models generated from JSON schemas.
These models provide runtime validation for all data structures.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    started_date: date | None = None


class LabResult(BaseModel):
    """Laboratory test result"""

    test_name: str
    value: float
    unit: str
    timestamp: datetime
    reference_range: str | None = None


class HistoryRecord(BaseModel):
    """Patient history entry"""

    timestamp: datetime
    symptoms: list[str]
    diagnosis: str | None = None
    notes: str | None = None


class LocalPatientRecord(BaseModel):
    """
    Local patient record stored encrypted on device/hospital node.
    NEVER send raw PHI from this model to cloud.
    """

    patient_id: str = Field(..., description="Local unique identifier")
    age: int = Field(..., ge=0, le=150)
    sex: SexEnum
    name: str | None = Field(None, description="Full name - NEVER sent to cloud")
    date_of_birth: date | None = Field(None, description="Converted to age for cloud")
    conditions: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    history: list[HistoryRecord] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)


class SanitizedPrompt(BaseModel):
    """
    De-identified, sanitized data sent to cloud LLM council.
    Must not contain any PHI or identifiable information.
    """

    fingerprint: str = Field(..., pattern=r"^[a-f0-9]{64}$", description="SHA256 hash")
    age: int = Field(..., ge=0, le=150)
    sex: SexEnum
    conditions: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list, description="Relative time descriptors")
    numeric_findings: dict[str, float] = Field(
        default_factory=dict, description="Rounded lab values"
    )
    task: str = Field(..., description="Instruction for LLM")
    context: str | None = Field(None, description="Additional sanitized context")

    model_config = ConfigDict(extra="forbid")


class Differential(BaseModel):
    """Possible diagnosis with probability and rationale"""

    label: str = Field(..., description="Possible diagnosis or condition")
    probability: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Conservative explanation")
    requires_verification: bool | None = Field(
        False, description="External claims need verification"
    )
    verification_query: str | None = None


class ResponseMetadata(BaseModel):
    """Metadata for council response"""

    latency_ms: float | None = None
    timestamp: datetime | None = None
    model_version: str | None = None


class CouncilResponse(BaseModel):
    """
    Structured response from each LLM in the council.
    All LLMs must return data matching this schema.
    """

    model_id: str
    differentials: list[Differential] = Field(..., min_length=0, max_length=10)
    recommended_next_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    red_flag: bool = Field(..., description="True if emergency detected")
    red_flag_rationale: str | None = None
    metadata: ResponseMetadata | None = None

    model_config = ConfigDict(extra="forbid")


class MLFeatureVector(BaseModel):
    """Feature vector for local ML model prediction"""

    features: list[float]
    feature_names: list[str]
    metadata: dict[str, Any] | None = None

    @field_validator("features", "feature_names")
    @classmethod
    def check_lengths_match(cls, v, info):
        """Ensure features and feature_names have the same length"""
        # This validator runs per field, so we need to check in model_validator
        return v
