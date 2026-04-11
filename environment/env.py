"""
Voice Authenticity Detection Environment — 5-action multi-step agent loop.

Actions:
    request_temporal_features  — reveals jitter, shimmer, HNR
    request_spectral_features  — reveals MFCC values
    request_comparison         — returns similarity to real/fake reference centroids
    analyze_evidence           — synthesizes accumulated evidence
    final_classify             — submits label + confidence + reasoning (terminal)

Partial observability: the agent starts with NO features visible and must
actively query the environment to build its picture before classifying.

Step-level rewards provide shaping signals throughout the episode.
"""

import numpy as np
import random
from typing import List, Dict, Optional, Any
from environment.models import VoiceObservation, ActionType

# ── Task registry ───────────────────────────────────────────────────────

TASKS = [
    "clean_detection",
    "compressed_detection",
    "adversarial_detection",
    "streaming_detection",
    "phonecall_detection",
]

DIFFICULTY_MAP = {
    "clean_detection":       "easy",
    "compressed_detection":  "medium",
    "adversarial_detection": "hard",
    "streaming_detection":   "medium_hard",
    "phonecall_detection":   "extreme",
}

DATA_FILES = {
    "clean_detection": (
        "environment/data/features.npy",
        "environment/data/labels.npy",
    ),
    "compressed_detection": (
        "environment/data/features_compressed.npy",
        "environment/data/labels_compressed.npy",
    ),
    "adversarial_detection": (
        "environment/data/features_adversarial.npy",
        "environment/data/labels_adversarial.npy",
    ),
    "streaming_detection": (
        "environment/data/features_streaming.npy",
        "environment/data/labels_streaming.npy",
    ),
    "phonecall_detection": (
        "environment/data/features_phonecall.npy",
        "environment/data/labels_phonecall.npy",
    ),
}

MAX_STEPS = 6  # 5 actions + 1 buffer

# ── Step-level reward constants ─────────────────────────────────────────

REWARD_FIRST_ACTION_GATHER = 0.05        # first action is a feature request
REWARD_MULTI_FEATURE_TYPES = 0.05        # requested both temporal AND spectral
REWARD_ANALYZE_BEFORE_CLASSIFY = 0.05    # used analyze_evidence before final
PENALTY_JUMP_TO_CLASSIFY = -0.10         # final_classify as first action
PENALTY_REPEAT_ACTION = -0.05            # same action twice
PENALTY_CONTRADICTORY_REASONING = -0.10  # reasoning contradicts label


class VoiceAuthenticityEnv:
    """Multi-step voice authenticity detection environment.

    The agent starts with no features visible and must issue actions to
    reveal information before making a final classification.
    """

    def __init__(self, task_name: str = "clean_detection"):
        assert task_name in TASKS, f"Unknown task: {task_name}. Valid: {TASKS}"
        self.task_name = task_name
        self.difficulty = DIFFICULTY_MAP[task_name]

        feat_file, label_file = DATA_FILES[task_name]
        self.features = np.load(feat_file)
        self.labels = np.load(label_file)
        self.raw_features = np.load("environment/data/features_raw.npy")
        self.indices = list(range(len(self.labels)))

        # Precompute reference centroids for comparison action
        self._compute_reference_centroids()

        # Episode state
        self.current_idx: Optional[int] = None
        self.step_number: int = 0
        self.done: bool = False
        self.action_history: List[str] = []
        self.revealed_features: Dict[str, Any] = {}
        self.step_rewards: List[float] = []
        self.evidence_accumulated: List[str] = []
        self.np_random = np.random.default_rng(None)

        # Streaming task noise schedule (soft-gating)
        self._streaming_noise_schedule = {
            1: 0.8,   # very noisy early
            2: 0.5,
            3: 0.3,
            4: 0.1,
            5: 0.05,  # nearly clean late
        }

    def _compute_reference_centroids(self):
        """Compute mean feature vectors for real vs fake samples."""
        real_mask = self.labels == 0
        fake_mask = self.labels == 1

        if real_mask.sum() > 0:
            self.real_centroid = self.features[real_mask].mean(axis=0)
        else:
            self.real_centroid = np.full(self.features.shape[1], 0.05)

        if fake_mask.sum() > 0:
            self.fake_centroid = self.features[fake_mask].mean(axis=0)
        else:
            self.fake_centroid = np.full(self.features.shape[1], 0.05)

    def reset(self, seed: Optional[int] = None) -> VoiceObservation:
        """Reset episode state and select a new sample.

        Randomized per episode:
            - current_idx: which sample from the dataset is used (via RNG)
            - All episode-tracking state is cleared (action history, revealed
              features, step rewards, evidence)

        NOT randomized (fixed per task):
            - Feature data (features.npy, labels.npy) — loaded once at init
            - Reference centroids (real_centroid, fake_centroid) — precomputed
            - Difficulty level and task name — fixed by constructor
            - Streaming noise schedule — deterministic per step number

        Args:
            seed: Optional RNG seed for reproducibility. If provided, the
                  sample selection is deterministic for that seed.

        Returns:
            VoiceObservation with NO features visible (all zeroed to 0.05).
        """
        self.step_number = 0
        self.done = False
        self.action_history = []
        self.revealed_features = {}
        self.step_rewards = []
        self.evidence_accumulated = []
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            
        self.current_idx = int(self.np_random.integers(0, len(self.labels)))
        return self._make_observation()

    def step(self, action: dict) -> tuple:
        """Execute one action and return (observation, reward, done, info).

        Args:
            action: dict with 'action_type' and optionally label/confidence/reasoning.

        Returns:
            (VoiceObservation, float, bool, dict)
        """
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        action_type = action.get("action_type", "final_classify")

        # Validate action type
        valid_actions = [at.value for at in ActionType]
        if action_type not in valid_actions:
            raise ValueError(
                f"Unknown action_type: {action_type}. Valid: {valid_actions}"
            )

        # Track action
        self.action_history.append(action_type)
        self.step_number += 1

        # Compute step-level reward
        step_reward = self._compute_step_reward(action_type, action)

        # Dispatch to action handler
        if action_type == ActionType.REQUEST_TEMPORAL.value:
            obs, info = self._handle_request_temporal()
        elif action_type == ActionType.REQUEST_SPECTRAL.value:
            obs, info = self._handle_request_spectral()
        elif action_type == ActionType.REQUEST_COMPARISON.value:
            obs, info = self._handle_request_comparison()
        elif action_type == ActionType.ANALYZE_EVIDENCE.value:
            obs, info = self._handle_analyze_evidence(action)
        elif action_type == ActionType.FINAL_CLASSIFY.value:
            obs, final_reward, info = self._handle_final_classify(action)
            # Terminal action: reward is purely the grader score,
            # not inflated by step-level shaping bonuses.
            step_reward = final_reward

        self.step_rewards.append(step_reward)

        # Cap reward to strictly (0, 1) — never exactly 0.0 or 1.0
        step_reward = max(0.01, min(0.99, step_reward))

        # Check step limit
        if self.step_number >= MAX_STEPS and not self.done:
            self.done = True
            info["message"] = "Max steps reached. Episode ended."

        return obs, round(step_reward, 4), self.done, info

    def state(self) -> dict:
        """Return full environment state for debugging."""
        return {
            "task_name":         self.task_name,
            "difficulty":        self.difficulty,
            "step_number":       self.step_number,
            "done":              self.done,
            "current_idx":       self.current_idx,
            "action_history":    self.action_history,
            "revealed_features": list(self.revealed_features.keys()),
            "step_rewards":      self.step_rewards,
        }

    # ── Action handlers ─────────────────────────────────────────────────

    def _handle_request_temporal(self) -> tuple:
        """Reveal jitter, shimmer, HNR values."""
        raw = self.raw_features[self.current_idx]
        norm = self.features[self.current_idx]

        temporal_data = {
            "jitter":  round(float(raw[42]), 6),
            "shimmer": round(float(raw[43]), 6),
            "hnr":     round(float(raw[44]), 4),
            "jitter_normalized":  round(float(norm[42]), 4),
            "shimmer_normalized": round(float(norm[43]), 4),
            "hnr_normalized":     round(float(norm[44]), 4),
        }

        # Apply streaming noise if applicable
        if self.task_name == "streaming_detection":
            temporal_data = self._apply_streaming_noise(temporal_data)

        self.revealed_features["temporal"] = temporal_data
        self.evidence_accumulated.append(
            f"Temporal features: jitter={temporal_data['jitter']}, "
            f"shimmer={temporal_data['shimmer']}, hnr={temporal_data['hnr']}"
        )

        obs = self._make_observation()
        info = {
            "action": "request_temporal_features",
            "message": "Temporal features revealed: jitter, shimmer, HNR.",
            "data": temporal_data,
        }
        return obs, info

    def _handle_request_spectral(self) -> tuple:
        """Reveal MFCC mean and std values."""
        raw = self.raw_features[self.current_idx]
        norm = self.features[self.current_idx]

        spectral_data = {
            "mfcc_means": [round(float(v), 4) for v in raw[0:20]],
            "mfcc_stds":  [round(float(v), 4) for v in raw[20:40]],
            "zcr": round(float(raw[40]), 6),
            "spectral_centroid": round(float(raw[41]), 4),
            "mfcc_means_normalized": [round(float(v), 4) for v in norm[0:20]],
            "mfcc_stds_normalized":  [round(float(v), 4) for v in norm[20:40]],
        }

        # Apply streaming noise if applicable
        if self.task_name == "streaming_detection":
            spectral_data = self._apply_streaming_noise(spectral_data)

        self.revealed_features["spectral"] = spectral_data
        self.evidence_accumulated.append(
            f"Spectral features: {len(spectral_data['mfcc_means'])} MFCC coefficients, "
            f"ZCR={spectral_data['zcr']}, centroid={spectral_data['spectral_centroid']}"
        )

        obs = self._make_observation()
        info = {
            "action": "request_spectral_features",
            "message": "Spectral features revealed: 20 MFCC means, 20 MFCC stds, ZCR, spectral centroid.",
            "data": spectral_data,
        }
        return obs, info

    def _handle_request_comparison(self) -> tuple:
        """Compare this sample to known real/fake reference centroids."""
        sample = self.features[self.current_idx]

        # Cosine similarity to real and fake centroids
        real_sim = self._cosine_similarity(sample, self.real_centroid)
        fake_sim = self._cosine_similarity(sample, self.fake_centroid)

        # Euclidean distance
        real_dist = float(np.linalg.norm(sample - self.real_centroid))
        fake_dist = float(np.linalg.norm(sample - self.fake_centroid))

        comparison_data = {
            "cosine_similarity_to_real": round(real_sim, 4),
            "cosine_similarity_to_fake": round(fake_sim, 4),
            "euclidean_distance_to_real": round(real_dist, 4),
            "euclidean_distance_to_fake": round(fake_dist, 4),
            "closer_to": "real" if real_dist < fake_dist else "fake",
            "similarity_differential": round(real_sim - fake_sim, 4),
        }

        self.revealed_features["comparison"] = comparison_data
        self.evidence_accumulated.append(
            f"Comparison: cosine_sim_real={comparison_data['cosine_similarity_to_real']}, "
            f"cosine_sim_fake={comparison_data['cosine_similarity_to_fake']}, "
            f"closer_to={comparison_data['closer_to']}"
        )

        obs = self._make_observation()
        info = {
            "action": "request_comparison",
            "message": "Comparison to reference centroids computed.",
            "data": comparison_data,
        }
        return obs, info

    def _handle_analyze_evidence(self, action: dict) -> tuple:
        """Synthesize all gathered evidence into a structured summary."""
        evidence_parts = []

        # Build evidence summary from what's been revealed
        if "temporal" in self.revealed_features:
            t = self.revealed_features["temporal"]
            jitter_val = t.get("jitter", 0)
            shimmer_val = t.get("shimmer", 0)
            hnr_val = t.get("hnr", 0)

            # Provide interpretive guidance based on actual values
            jitter_interp = "elevated (typical of real speech)" if jitter_val > 0.025 else "low (typical of synthetic)"
            shimmer_interp = "elevated (typical of real speech)" if shimmer_val > 0.10 else "low (typical of synthetic)"
            hnr_interp = "low (typical of real speech)" if hnr_val < 12.0 else "high (typical of synthetic)"

            evidence_parts.append(
                f"TEMPORAL: jitter={jitter_val} ({jitter_interp}), "
                f"shimmer={shimmer_val} ({shimmer_interp}), "
                f"HNR={hnr_val} ({hnr_interp})"
            )

        if "spectral" in self.revealed_features:
            s = self.revealed_features["spectral"]
            mfcc_mean_avg = np.mean(s.get("mfcc_means", [0])) if s.get("mfcc_means") else 0
            mfcc_std_avg = np.mean(s.get("mfcc_stds", [0])) if s.get("mfcc_stds") else 0
            evidence_parts.append(
                f"SPECTRAL: avg_mfcc_mean={mfcc_mean_avg:.3f}, "
                f"avg_mfcc_std={mfcc_std_avg:.3f}, "
                f"zcr={s.get('zcr', 0)}, centroid={s.get('spectral_centroid', 0)}"
            )

        if "comparison" in self.revealed_features:
            c = self.revealed_features["comparison"]
            evidence_parts.append(
                f"COMPARISON: closer_to={c['closer_to']}, "
                f"diff={c['similarity_differential']}"
            )

        if not evidence_parts:
            summary = "No evidence gathered yet. Request features before analyzing."
        else:
            # Count evidence signals pointing to real vs fake
            real_signals = 0
            fake_signals = 0

            if "temporal" in self.revealed_features:
                t = self.revealed_features["temporal"]
                if t.get("jitter", 0) > 0.025:
                    real_signals += 1
                else:
                    fake_signals += 1
                if t.get("shimmer", 0) > 0.10:
                    real_signals += 1
                else:
                    fake_signals += 1
                if t.get("hnr", 0) < 12.0:
                    real_signals += 1
                else:
                    fake_signals += 1

            if "comparison" in self.revealed_features:
                c = self.revealed_features["comparison"]
                if c["closer_to"] == "real":
                    real_signals += 1
                else:
                    fake_signals += 1

            total_signals = real_signals + fake_signals
            if total_signals > 0:
                suggested_confidence = max(real_signals, fake_signals) / total_signals
                leaning = "REAL" if real_signals > fake_signals else "SYNTHETIC"
            else:
                suggested_confidence = 0.55
                leaning = "UNCERTAIN"

            # Adjust confidence for difficulty
            if self.difficulty in ("hard", "extreme", "medium_hard"):
                suggested_confidence = min(suggested_confidence, 0.80)

            summary = (
                f"Evidence analysis ({len(evidence_parts)} sources):\n"
                + "\n".join(f"  • {p}" for p in evidence_parts)
                + f"\n\nSignal tally: {real_signals} real vs {fake_signals} synthetic"
                + f"\nPreliminary assessment: leaning {leaning}"
                + f"\nSuggested confidence: {suggested_confidence:.2f}"
                + f"\nDifficulty context: {self.difficulty}"
            )

        self.revealed_features["analysis"] = {
            "summary": summary,
            "evidence_count": len(evidence_parts),
        }

        obs = self._make_observation(evidence_summary=summary)
        info = {
            "action": "analyze_evidence",
            "message": "Evidence synthesized.",
            "summary": summary,
            "evidence_count": len(evidence_parts),
        }
        return obs, info

    def _handle_final_classify(self, action: dict) -> tuple:
        """Submit final classification. Triggers grading. Episode ends."""
        from environment.graders import grade

        true_label = int(self.labels[self.current_idx])

        result = grade(
            true_label=true_label,
            action=action,
            difficulty=self.difficulty,
            action_history=self.action_history,
        )

        self.done = True

        obs = self._make_observation()
        info = {
            "action": "final_classify",
            "phase": "done",
            "true_label": true_label,
            "predicted_label": action.get("label", 0),
            "difficulty": self.difficulty,
            "task": self.task_name,
            "grader_breakdown": result["breakdown"],
            "grader_weights": result["weights"],
            "penalties": result["penalties"],
            "correct": result["correct"],
            "episode_summary": {
                "actions_taken": self.action_history,
                "features_revealed": list(self.revealed_features.keys()),
                "total_steps": self.step_number
            }
        }

        return obs, result["score"], info

    # ── Step-level reward computation ───────────────────────────────────

    def _compute_step_reward(self, action_type: str, action: dict) -> float:
        """Compute shaping reward for this step.

        Returns a value in [0.02, 0.18] — never exactly 0.0 or 1.0.
        The final clamp in step() further constrains to [0.05, 0.95].
        """
        reward = 0.05

        gathering_actions = {
            ActionType.REQUEST_TEMPORAL.value,
            ActionType.REQUEST_SPECTRAL.value,
            ActionType.REQUEST_COMPARISON.value,
        }

        # Reward: first action is a feature request
        if len(self.action_history) == 1 and action_type in gathering_actions:
            reward += REWARD_FIRST_ACTION_GATHER

        # Penalty: jumping straight to final_classify
        if len(self.action_history) == 1 and action_type == ActionType.FINAL_CLASSIFY.value:
            reward += PENALTY_JUMP_TO_CLASSIFY

        # Reward: multi-feature-type requests
        history_set = set(self.action_history)
        if (ActionType.REQUEST_TEMPORAL.value in history_set and
                ActionType.REQUEST_SPECTRAL.value in history_set and
                len(self.action_history) >= 2 and
                action_type in {ActionType.REQUEST_TEMPORAL.value, ActionType.REQUEST_SPECTRAL.value}):
            # Only award once: check if this is the action that completed the pair
            prev_set = set(self.action_history[:-1])
            if not (ActionType.REQUEST_TEMPORAL.value in prev_set and
                    ActionType.REQUEST_SPECTRAL.value in prev_set):
                reward += REWARD_MULTI_FEATURE_TYPES

        # Reward: analyze_evidence before final_classify
        if (action_type == ActionType.FINAL_CLASSIFY.value and
                ActionType.ANALYZE_EVIDENCE.value in self.action_history[:-1]):
            reward += REWARD_ANALYZE_BEFORE_CLASSIFY

        # Penalty: repeating same action
        if len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            reward += PENALTY_REPEAT_ACTION

        # Penalty: contradictory reasoning (only on final_classify)
        if action_type == ActionType.FINAL_CLASSIFY.value:
            label = action.get("label", 0)
            reasoning = action.get("reasoning", "").lower()
            if label == 0 and any(kw in reasoning for kw in ["synthetic", "fake", "artificial", "generated"]):
                if not any(kw in reasoning for kw in ["not synthetic", "not fake", "not artificial"]):
                    reward += PENALTY_CONTRADICTORY_REASONING
            elif label == 1 and any(kw in reasoning for kw in ["real", "human", "natural", "authentic"]):
                if not any(kw in reasoning for kw in ["not real", "not human", "not natural"]):
                    reward += PENALTY_CONTRADICTORY_REASONING

        # Ensure shaping reward never produces exactly 0.0 or 1.0
        return max(0.02, min(0.18, reward))

    # ── Streaming noise (soft-gating) ───────────────────────────────────

    def _apply_streaming_noise(self, data: dict) -> dict:
        """Apply noise to features based on step number for streaming task.

        Earlier steps get noisier data, later steps get cleaner data.
        This is soft-gating: features are always available but with
        varying fidelity.
        """
        noise_level = self._streaming_noise_schedule.get(
            self.step_number, 0.05
        )

        noisy_data = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, noise_level * abs(value) + 1e-6)
                noisy_data[key] = round(float(value + noise), 6)
            elif isinstance(value, list):
                noisy_data[key] = [
                    round(float(v + np.random.normal(0, noise_level * abs(v) + 1e-6)), 4)
                    for v in value
                ]
            else:
                noisy_data[key] = value

        return noisy_data

    # ── Helper methods ──────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.05
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _make_observation(
        self,
        evidence_summary: Optional[str] = None,
    ) -> VoiceObservation:
        """Build observation from current state.

        The full feature vector is only included if the agent has requested
        both temporal AND spectral features, or if the episode is done.
        Otherwise it's zeroed out to enforce partial observability.
        """
        has_temporal = "temporal" in self.revealed_features
        has_spectral = "spectral" in self.revealed_features
        full_revealed = has_temporal and has_spectral

        if full_revealed or self.done:
            feat = self.features[self.current_idx].tolist()
        else:
            # Partial observability: base value 0.05 for unrevealed features
            feat = [0.05] * self.features.shape[1]

        # Build hint based on current state
        hint = self._build_hint()

        # Comparison result from revealed features
        comparison = self.revealed_features.get("comparison", None)

        # Available actions
        available = self._get_available_actions()

        return VoiceObservation(
            features=feat,
            task_name=self.task_name,
            step_number=self.step_number,
            difficulty=self.difficulty,
            sample_id=int(self.current_idx),
            hint=hint,
            visible_features=dict(self.revealed_features),
            evidence_summary=evidence_summary,
            comparison_result=comparison,
            available_actions=available,
            actions_taken=list(self.action_history),
        )

    def _build_hint(self) -> str:
        """Build context hint for the agent."""
        if self.done:
            return "Episode complete."

        if self.step_number == 0:
            hint = (
                f"Task: {self.task_name} (difficulty: {self.difficulty}). "
                f"You have {MAX_STEPS - self.step_number} steps remaining. "
                "No features are visible yet. Use request_temporal_features, "
                "request_spectral_features, or request_comparison to gather "
                "evidence before classifying."
            )
            if self.difficulty in ("hard", "extreme"):
                hint += " Warning: this is a challenging task. Gather thorough evidence and calibrate your confidence carefully."
            if self.task_name == "streaming_detection":
                hint += " Note: this is a streaming scenario — earlier feature requests may contain noise that reduces over time."
            if self.task_name == "phonecall_detection":
                hint += " Note: this is a phone call scenario with heavy codec compression and background noise."
            return hint

        parts = [
            f"Step {self.step_number}/{MAX_STEPS}.",
            f"Task: {self.task_name} ({self.difficulty}).",
            f"Actions taken: {', '.join(self.action_history)}.",
        ]

        if self.revealed_features:
            revealed = list(self.revealed_features.keys())
            parts.append(f"Features revealed: {', '.join(revealed)}.")

        remaining = MAX_STEPS - self.step_number
        if remaining <= 2:
            parts.append(f"⚠️ Only {remaining} steps remaining — consider classifying soon.")

        return " ".join(parts)

    def _get_available_actions(self) -> List[str]:
        """Return list of actions the agent can still take."""
        if self.done:
            return []

        available = []
        for at in ActionType:
            # final_classify is always available
            if at == ActionType.FINAL_CLASSIFY:
                available.append(at.value)
                continue
            # Don't allow repeating the exact same action consecutively
            # (but allow re-requesting after other actions)
            if (self.action_history and
                    self.action_history[-1] == at.value):
                continue
            available.append(at.value)

        return available