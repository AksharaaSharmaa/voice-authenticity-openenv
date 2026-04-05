from pydantic import BaseModel, Field
from typing import Optional, List

class VoiceObservation(BaseModel):
    features: List[float]
    task_name: str
    step_number: int
    difficulty: str
    sample_id: int
    hint: Optional[str] = None  # extra context for hard task

class VoiceAction(BaseModel):
    label: int = Field(..., ge=0, le=1)          # 0=real, 1=synthetic
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="")

class VoiceReward(BaseModel):
    score: float
    correct: bool
    confidence_penalty: float
    breakdown: str