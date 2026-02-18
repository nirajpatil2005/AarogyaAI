"""
Mock council stub for development.
Simulates cloud LLM council responses.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Mock Council Stub")


class SanitizedPrompt(BaseModel):
    fingerprint: str
    age: int
    sex: str
    conditions: List[str]
    symptoms: List[str]
    task: str


class Differential(BaseModel):
    label: str
    probability: float
    rationale: str


class CouncilResponse(BaseModel):
    model_id: str
    differentials: List[Differential]
    recommended_next_steps: List[str]
    confidence: float
    red_flag: bool


@app.post("/council/gpt4")
async def gpt4_stub(prompt: SanitizedPrompt):
    """Mock GPT-4 response"""
    return CouncilResponse(
        model_id="mock-gpt4",
        differentials=[
            Differential(
                label="Common cold",
                probability=0.6,
                rationale="Symptoms suggest possible upper respiratory infection"
            )
        ],
        recommended_next_steps=[
            "Monitor symptoms",
            "Stay hydrated",
            "Rest",
            "Contact provider if symptoms worsen"
        ],
        confidence=0.7,
        red_flag=False
    )


@app.post("/council/claude")
async def claude_stub(prompt: SanitizedPrompt):
    """Mock Claude response"""
    return CouncilResponse(
        model_id="mock-claude",
        differentials=[
            Differential(
                label="Viral infection",
                probability=0.65,
                rationale="Pattern consistent with viral illness"
            )
        ],
        recommended_next_steps=[
            "Rest and hydration",
            "Monitor temperature",
            "Seek care if symptoms persist"
        ],
        confidence=0.72,
        red_flag=False
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mock-council"}
