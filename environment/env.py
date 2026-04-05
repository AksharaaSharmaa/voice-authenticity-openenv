import numpy as np
import random
from environment.models import VoiceObservation

TASKS = ["clean_detection", "compressed_detection", "adversarial_detection"]

DIFFICULTY_MAP = {
    "clean_detection":       "easy",
    "compressed_detection":  "medium",
    "adversarial_detection": "hard"
}

DATA_FILES = {
    "clean_detection": (
        "environment/data/features.npy",
        "environment/data/labels.npy"
    ),
    "compressed_detection": (
        "environment/data/features_compressed.npy",
        "environment/data/labels_compressed.npy"
    ),
    "adversarial_detection": (
        "environment/data/features_adversarial.npy",
        "environment/data/labels_adversarial.npy"
    ),
}

class VoiceAuthenticityEnv:
    def __init__(self, task_name: str = "clean_detection"):
        assert task_name in TASKS, f"Unknown task: {task_name}"
        self.task_name  = task_name
        self.difficulty = DIFFICULTY_MAP[task_name]

        feat_file, label_file = DATA_FILES[task_name]
        self.features     = np.load(feat_file)
        self.labels       = np.load(label_file)
        self.raw_features = np.load("environment/data/features_raw.npy")
        self.indices      = list(range(len(self.labels)))

        self.current_idx  = None
        self.step_number  = 0
        self.done         = False
        self.phase        = "analyze"  # analyze → decide
        self.focus_features = None

    def reset(self):
        self.step_number    = 0
        self.done           = False
        self.phase          = "analyze"
        self.focus_features = None
        self.current_idx    = random.choice(self.indices)
        return self._make_observation()

    def step(self, action: dict):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        # Phase 1 — agent requests focused analysis
        if self.phase == "analyze":
            self.focus_features = action.get("focus", ["jitter", "shimmer", "hnr"])
            self.step_number += 1
            self.phase = "decide"
            obs = self._make_observation()
            return obs, 0.0, False, {
                "phase": "decide",
                "message": "Analysis received. Now submit your final classification.",
                "focused_on": self.focus_features
            }

        # Phase 2 — agent submits final classification
        elif self.phase == "decide":
            from environment.graders import grade
            true_label = int(self.labels[self.current_idx])
            reward     = grade(true_label, action, self.difficulty)

            self.step_number += 1
            self.done  = True
            self.phase = "done"

            obs  = self._make_observation()
            info = {
                "phase":      "done",
                "true_label": true_label,
                "difficulty": self.difficulty,
                "task":       self.task_name
            }
            return obs, reward, self.done, info

    def state(self):
        return {
            "task_name":      self.task_name,
            "difficulty":     self.difficulty,
            "step_number":    self.step_number,
            "phase":          self.phase,
            "done":           self.done,
            "current_idx":    self.current_idx,
            "focus_features": self.focus_features
        }

    def _make_observation(self) -> VoiceObservation:
        feat = self.features[self.current_idx].tolist()
        raw  = self.raw_features[self.current_idx]

        if self.phase == "analyze":
            hint = f"Phase 1 of 2: Request which features to focus on by returning focus=[list of feature names]. Available: jitter, shimmer, hnr, mfcc, spectral. | Raw values → jitter={raw[42]:.5f} shimmer={raw[43]:.5f} hnr={raw[44]:.4f}"
        elif self.phase == "decide":
            focused = self.focus_features or ["jitter", "shimmer", "hnr"]
            hint = f"Phase 2 of 2: Submit your final classification. You focused on: {focused}. | Raw values → jitter={raw[42]:.5f} shimmer={raw[43]:.5f} hnr={raw[44]:.4f}"
            if self.difficulty == "medium":
                hint += " | Note: audio has been codec-compressed."
            elif self.difficulty == "hard":
                hint += " | Warning: adversarial sample — feature distributions overlap with real speech."
        else:
            hint = "Episode complete."

        return VoiceObservation(
            features    = feat,
            task_name   = self.task_name,
            step_number = self.step_number,
            difficulty  = self.difficulty,
            sample_id   = int(self.current_idx),
            hint        = hint
        )