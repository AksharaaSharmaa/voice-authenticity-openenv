import numpy as np
import random
from environment.models import VoiceObservation

TASKS = ["clean_detection", "compressed_detection", "adversarial_detection"]

DIFFICULTY_MAP = {
    "clean_detection":      "easy",
    "compressed_detection": "medium",
    "adversarial_detection":"hard"
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
        self.features = np.load(feat_file)
        self.labels   = np.load(label_file)
        
        # Load raw features for interpretable key values
        self.raw_features = np.load("environment/data/features_raw.npy")
        
        self.indices = list(range(len(self.labels)))
        self.current_idx = None
        self.step_number = 0
        self.done = False
        self.max_steps = 1

    def reset(self):
        self.step_number = 0
        self.done        = False
        self.current_idx = random.choice(self.indices)
        return self._make_observation()

    def step(self, action: dict):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        from environment.graders import grade
        true_label = int(self.labels[self.current_idx])
        reward     = grade(true_label, action, self.difficulty)

        self.step_number += 1
        self.done = True

        obs  = self._make_observation()
        info = {
            "true_label": true_label,
            "difficulty": self.difficulty,
            "task":       self.task_name
        }
        return obs, reward, self.done, info

    def state(self):
        return {
            "task_name":   self.task_name,
            "difficulty":  self.difficulty,
            "step_number": self.step_number,
            "done":        self.done,
            "current_idx": self.current_idx
        }

    def _make_observation(self) -> VoiceObservation:
        feat = self.features[self.current_idx].tolist()
        raw  = self.raw_features[self.current_idx]
        
        hint = None
        if self.difficulty == "medium":
            hint = "Audio has been codec-compressed. Features may be degraded."
        elif self.difficulty == "hard":
            hint = "Warning: adversarial sample — synthetic audio crafted to mimic real speech."

        return VoiceObservation(
            features    = feat,
            task_name   = self.task_name,
            step_number = self.step_number,
            difficulty  = self.difficulty,
            sample_id   = int(self.current_idx),
            hint        = (hint or "") + f" | Key values → jitter={raw[42]:.5f} shimmer={raw[43]:.5f} hnr={raw[44]:.4f}"
        )