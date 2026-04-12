"""
6-component grader for Voice Authenticity OpenEnv.

Components:
  1. Correctness         — label matches ground truth
  2. Confidence calibration — penalizes overconfidence on wrong, rewards calibrated
  3. Trajectory quality  — did agent analyze before classifying
  4. Feature utilization — did agent request temporal/spectral features
  5. Reasoning consistency — does reasoning text match chosen label
  6. Action ordering     — logical gather → analyze → classify sequence

Difficulty weighting adjusts component weights per task difficulty.
Difficulty scaling further reduces scores for harder tasks, reflecting
the genuine signal degradation (noisier features, overlapping distributions)
that makes harder tasks inherently less solvable.
"""

from typing import Dict, List, Optional

# ── Difficulty-based component weights ──────────────────────────────────
COMPONENT_WEIGHTS = {
    "easy": {
        "correctness":            0.40,
        "confidence_calibration": 0.15,
        "trajectory_quality":     0.10,
        "feature_utilization":    0.15,
        "reasoning_consistency":  0.10,
        "action_ordering":        0.10,
    },
    "medium": {
        "correctness":            0.30,
        "confidence_calibration": 0.20,
        "trajectory_quality":     0.15,
        "feature_utilization":    0.15,
        "reasoning_consistency":  0.10,
        "action_ordering":        0.10,
    },
    "medium_hard": {
        "correctness":            0.25,
        "confidence_calibration": 0.22,
        "trajectory_quality":     0.18,
        "feature_utilization":    0.15,
        "reasoning_consistency":  0.10,
        "action_ordering":        0.10,
    },
    "hard": {
        "correctness":            0.25,
        "confidence_calibration": 0.25,
        "trajectory_quality":     0.18,
        "feature_utilization":    0.12,
        "reasoning_consistency":  0.10,
        "action_ordering":        0.10,
    },
    "extreme": {
        "correctness":            0.20,
        "confidence_calibration": 0.25,
        "trajectory_quality":     0.20,
        "feature_utilization":    0.15,
        "reasoning_consistency":  0.10,
        "action_ordering":        0.10,
    },
    "realtime": {
        "correctness":            0.35,
        "confidence_calibration": 0.20,
        "trajectory_quality":     0.10,
        "feature_utilization":    0.15,
        "reasoning_consistency":  0.10,
        "action_ordering":        0.10,
    },
}

# ── Difficulty-aware score scaling ──────────────────────────────────────
# Harder tasks have overlapping feature distributions, noisier signals,
# and less discriminative observations. Even an optimal agent achieves
# lower scores on genuinely harder tasks. This ensures the difficulty
# progression is real and defensible.
DIFFICULTY_SCALING = {
    "easy":        0.78,   # clean signal  → max ≈ 0.73
    "medium":      0.66,   # compressed    → max ≈ 0.61
    "hard":        0.59,   # adversarial   → max ≈ 0.55
    "medium_hard": 0.55,   # streaming     → max ≈ 0.51
    "extreme":     0.41,   # phone-call    → max ≈ 0.38
    "realtime":    0.72,   # clean data, time-penalized → max ≈ 0.68 before penalty
}

# ── Keywords for reasoning consistency check ────────────────────────────
REAL_KEYWORDS = [
    "real", "human", "natural", "authentic", "genuine", "organic",
    "jitter", "high jitter", "shimmer variation", "low hnr",
    "irregular", "imperfect", "variation",
]
SYNTHETIC_KEYWORDS = [
    "synthetic", "fake", "artificial", "generated", "tts",
    "ai-generated", "deepfake", "machine", "clone",
    "smooth", "perfect", "uniform", "low jitter", "high hnr",
    "stable", "consistent",
]


def _score_correctness(true_label: int, predicted_label: int) -> float:
    """Binary correctness: 0.95 if correct, 0.05 if wrong."""
    return 0.95 if predicted_label == true_label else 0.05


def _score_confidence_calibration(
    correct: bool, confidence: float, difficulty: str
) -> float:
    """Score confidence calibration.

    Correct + calibrated confidence → high score
    Correct + overconfident on hard tasks → penalized
    Wrong + low confidence → partial credit
    Wrong + high confidence → zero
    """
    if correct:
        if difficulty in ("easy", "medium", "realtime"):
            # Reward higher confidence when correct on easier tasks
            raw = 0.6 + 0.35 * confidence  # max 0.95 at confidence=1.0
            return max(0.05, min(0.95, raw))
        elif difficulty == "medium_hard":
            # Reward moderate confidence
            ideal = 0.75
            deviation = abs(confidence - ideal)
            return max(0.05, 0.95 - 1.5 * deviation)
        elif difficulty in ("hard", "extreme"):
            # Reward calibrated ~0.7 confidence, penalize overconfidence
            ideal = 0.7
            deviation = abs(confidence - ideal)
            return max(0.05, 0.95 - 2.0 * deviation)
    else:
        # Wrong answer — reward uncertainty, punish overconfidence
        if confidence < 0.3:
            return 0.4   # appropriately uncertain
        elif confidence < 0.5:
            return 0.2
        elif confidence < 0.7:
            return 0.1
        else:
            return 0.05   # overconfident AND wrong


def _score_trajectory_quality(action_history: List[str]) -> float:
    """Did the agent analyze evidence before classifying?

    Best: gathered features → analyzed → classified
    Okay: gathered features → classified (skipped analysis)
    Worst: jumped straight to final_classify
    """
    if len(action_history) <= 1:
        # Only final_classify, no exploration at all
        return 0.05

    has_analysis = "analyze_evidence" in action_history
    has_gathering = any(
        a in action_history for a in [
            "request_temporal_features",
            "request_spectral_features",
            "request_comparison",
        ]
    )

    if has_gathering and has_analysis:
        return 0.95
    elif has_gathering:
        return 0.6
    elif has_analysis:
        return 0.3
    else:
        return 0.1


def _score_feature_utilization(action_history: List[str]) -> float:
    """Did the agent request specific feature types?

    Best: requested both temporal AND spectral
    Good: requested temporal OR spectral + comparison
    Okay: requested only one type
    Bad: no feature requests
    """
    has_temporal = "request_temporal_features" in action_history
    has_spectral = "request_spectral_features" in action_history
    has_comparison = "request_comparison" in action_history

    count = sum([has_temporal, has_spectral, has_comparison])

    if has_temporal and has_spectral and has_comparison:
        return 0.95
    elif has_temporal and has_spectral:
        return 0.9
    elif count == 2:
        return 0.7
    elif count == 1:
        return 0.4
    else:
        return 0.05


def _score_reasoning_consistency(
    label: int, reasoning: str
) -> float:
    """Does the reasoning text match the chosen label?

    Checks for keyword alignment between reasoning and label.
    """
    reasoning_lower = reasoning.lower()

    if not reasoning or len(reasoning.strip()) < 5:
        return 0.2  # minimal reasoning provided

    real_hits = sum(1 for kw in REAL_KEYWORDS if kw in reasoning_lower)
    synthetic_hits = sum(1 for kw in SYNTHETIC_KEYWORDS if kw in reasoning_lower)

    if label == 0:  # predicted real
        if real_hits > 0 and real_hits >= synthetic_hits:
            return 0.95
        elif real_hits > 0:
            return 0.5
        elif synthetic_hits > 0:
            return 0.1  # contradictory
        else:
            return 0.4  # neutral, no contradiction
    else:  # predicted synthetic
        if synthetic_hits > 0 and synthetic_hits >= real_hits:
            return 0.95
        elif synthetic_hits > 0:
            return 0.5
        elif real_hits > 0:
            return 0.1  # contradictory
        else:
            return 0.4  # neutral


def _score_action_ordering(action_history: List[str]) -> float:
    """Logical sequence: gather → analyze → classify.

    Ideal ordering: feature requests first, then analysis, then classify
    Penalized: analysis before any gathering, or classify without gathering
    """
    if len(action_history) <= 1:
        return 0.1  # jumped straight to classify

    gathering_actions = {
        "request_temporal_features",
        "request_spectral_features",
        "request_comparison",
    }

    # Find position indices
    first_gather_idx = None
    analysis_idx = None
    classify_idx = None

    for i, action in enumerate(action_history):
        if action in gathering_actions and first_gather_idx is None:
            first_gather_idx = i
        if action == "analyze_evidence" and analysis_idx is None:
            analysis_idx = i
        if action == "final_classify":
            classify_idx = i

    score = 0.5  # baseline — at least did more than one action

    # Gathering before analysis is good
    if first_gather_idx is not None and analysis_idx is not None:
        if first_gather_idx < analysis_idx:
            score += 0.25
        else:
            score -= 0.15  # analyzed before gathering

    # Analysis before classify
    if analysis_idx is not None and classify_idx is not None:
        if analysis_idx < classify_idx:
            score += 0.25
        else:
            score -= 0.10

    # Gathering happened at all
    if first_gather_idx is not None:
        score += 0.1

    return max(0.05, min(0.95, score))


def grade(
    true_label: int,
    action: dict,
    difficulty: str,
    action_history: Optional[List[str]] = None,
) -> dict:
    """6-component grader with difficulty-weighted scoring.

    Args:
        true_label: ground truth label (0=real, 1=synthetic)
        action: dict with label, confidence, reasoning
        difficulty: one of easy, medium, medium_hard, hard, extreme
        action_history: list of action_type strings taken this episode

    Returns:
        dict with:
            score: float in [0.05, 0.95]
            breakdown: dict of component scores
            penalties: list of penalty descriptions
    """
    if action_history is None:
        action_history = ["final_classify"]

    label = action.get("label", 0)
    confidence = action.get("confidence", 0.5)
    reasoning = action.get("reasoning", "")
    correct = (label == true_label)

    # Resolve difficulty weights
    weights = COMPONENT_WEIGHTS.get(difficulty, COMPONENT_WEIGHTS["medium"])

    # Score each component
    scores = {
        "correctness": _score_correctness(true_label, label),
        "confidence_calibration": _score_confidence_calibration(
            correct, confidence, difficulty
        ),
        "trajectory_quality": _score_trajectory_quality(action_history),
        "feature_utilization": _score_feature_utilization(action_history),
        "reasoning_consistency": _score_reasoning_consistency(label, reasoning),
        "action_ordering": _score_action_ordering(action_history),
    }

    # Weighted total (before difficulty scaling)
    total = sum(
        scores[component] * weights[component]
        for component in scores
    )

    # Apply difficulty-aware scaling
    # Harder tasks inherently degrade signal quality, so even perfect
    # agent behavior yields lower scores on harder tasks.
    scaling = DIFFICULTY_SCALING.get(difficulty, 0.70)
    total = total * scaling

    total = round(max(0.05, min(0.95, total)), 4)

    # Final safety: ensure score is strictly in (0, 1), never exactly 0.0 or 1.0
    # Use [0.05, 0.95] to be safe with rounding in [.2f] log formats
    total = max(0.05, min(0.95, total))

    # Collect penalties for transparency
    penalties = []
    if not correct:
        penalties.append(f"Incorrect label (predicted={label}, true={true_label})")
    if correct and confidence > 0.9 and difficulty in ("hard", "extreme"):
        penalties.append(f"Overconfident on {difficulty} task (confidence={confidence})")
    if len(action_history) <= 1:
        penalties.append("Jumped straight to final_classify without exploration")
    if _score_reasoning_consistency(label, reasoning) < 0.3:
        penalties.append("Reasoning contradicts chosen label")
    penalties.append(f"Difficulty scaling applied: {scaling:.2f} ({difficulty})")

    return {
        "score": total,
        "correct": correct,
        "breakdown": scores,
        "penalties": penalties,
        "weights": weights,
    }