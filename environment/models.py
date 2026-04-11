from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    """Five distinct agent actions for real partial observability."""
    REQUEST_TEMPORAL = "request_temporal_features"
    REQUEST_SPECTRAL = "request_spectral_features"
    REQUEST_COMPARISON = "request_comparison"
    ANALYZE_EVIDENCE = "analyze_evidence"
    FINAL_CLASSIFY = "final_classify"


class VoiceObservation(BaseModel):
    """Observation returned to the agent after each action.

    features: full 48-dim vector (only populated after sufficient exploration
              or on final step)
    visible_features: dict of feature groups revealed so far
    evidence_summary: structured summary from analyze_evidence action
    comparison_result: similarity scores from request_comparison action
    """
    features: List[float]
    task_name: str
    step_number: int
    difficulty: str
    sample_id: int
    hint: Optional[str] = None
    visible_features: Dict[str, Any] = Field(default_factory=dict)
    evidence_summary: Optional[str] = None
    comparison_result: Optional[Dict[str, Any]] = None
    available_actions: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)


class VoiceAction(BaseModel):
    """Action submitted by the agent.

    action_type: which of the 5 actions to perform
    label: classification (only used for final_classify)
    confidence: agent confidence (used for final_classify and analyze_evidence)
    reasoning: explanation (used for final_classify)
    focus: optional list of feature names (backward compat)
    """
    action_type: str = Field(default="final_classify")
    label: int = Field(default=0, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    focus: List[str] = Field(default_factory=list)


class GraderBreakdown(BaseModel):
    """Detailed 6-component grading breakdown."""
    correctness: float = 0.0
    confidence_calibration: float = 0.0
    trajectory_quality: float = 0.0
    feature_utilization: float = 0.0
    reasoning_consistency: float = 0.0
    action_ordering: float = 0.0


class VoiceReward(BaseModel):
    """Reward with full breakdown."""
    score: float
    correct: bool
    step_rewards: List[float] = Field(default_factory=list)
    grader_breakdown: Optional[GraderBreakdown] = None
    penalties: List[str] = Field(default_factory=list)
    breakdown: str = ""