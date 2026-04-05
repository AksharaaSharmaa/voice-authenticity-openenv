---
title: Voice Authenticity OpenEnv
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🎙️ Voice Authenticity Detection — OpenEnv Environment

An reinforcement learning environment for training and evaluating AI agents to detect synthetic (AI-generated) speech across real-world degradation conditions.

> Voice fraud is a growing crisis. This environment trains agents to detect synthetic speech under clean, compressed, and adversarial conditions — directly applicable to fraud detection, content moderation, and voice authentication systems.

---

## 🌍 Real-World Motivation

AI-generated voices (ElevenLabs, Coqui, etc.) are increasingly used for:
- Phone fraud and social engineering attacks
- Deepfake audio in misinformation campaigns
- Identity spoofing in voice authentication systems

This environment provides a structured benchmark for training agents to detect synthetic speech under realistic degradation conditions that existing classifiers struggle with.

---

## 🏗️ Environment Overview

The environment serves 48-dimensional feature vectors extracted from audio samples. Agents must classify each sample as real or synthetic, with a confidence score that reflects genuine uncertainty.

### Why Feature Vectors, Not Raw Audio?
- Fits within 2 vCPU / 8GB RAM constraints
- Feature extraction done offline — inference is fast
- Enables LLM-native reasoning over audio characteristics, which is not possible with raw waveforms under current infrastructure constraints
- Interpretable observations for LLM-based agents

### Dataset
- Real speech: 250 samples from `garystafford/deepfake-audio-detection` (authentic human recordings)
- Synthetic speech: 250 samples (ElevenLabs, Hume AI, and other TTS platforms)
- Total: 500 labeled samples across 3 task variants
- Dataset focuses on evaluation structure, not scale — the feature pipeline supports arbitrary dataset scaling for production use

---

## 📐 Observation Space

Each observation is a 48-dimensional float32 vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0–19 | MFCC means | Timbre and spectral shape of the voice |
| 20–39 | MFCC std devs | Temporal variation in spectral characteristics |
| 40 | Zero crossing rate | Signal sign changes per frame |
| 41 | Spectral centroid | Brightness of the sound |
| 42 | Jitter | Cycle-to-cycle frequency instability |
| 43 | Shimmer | Amplitude variation between glottal pulses |
| 44 | HNR | Ratio of harmonic energy to background noise |
| 45–47 | Compression artifacts | Spectral bandwidth, rolloff, RMS energy |

### Key Discriminating Features

These acoustic properties capture fundamental differences between human and synthetic vocal production:

- Jitter: measures cycle-to-cycle frequency instability in the voice signal
- Shimmer: tracks amplitude variation between consecutive glottal pulses
- HNR: quantifies the ratio of harmonic energy to noise in the signal

### Observation Schema (Pydantic)
```python
class VoiceObservation(BaseModel):
    features: List[float]      # 48-dim feature vector (normalized)
    task_name: str             # current task
    step_number: int           # current step in episode
    difficulty: str            # easy | medium | hard
    sample_id: int             # index into dataset
    hint: Optional[str]        # task context and key raw values
```

---

## 🎯 Action Space
```python
class VoiceAction(BaseModel):
    label: int        # 0 = real, 1 = synthetic
    confidence: float # agent confidence in [0.0, 1.0]
    reasoning: str    # brief explanation of decision
```

---

## 🏆 Tasks

### Task 1 — Clean Detection (Easy)
- Description: Classify real vs synthetic speech from clean, unmodified audio features
- Difficulty: Easy
- Expected agent score: 0.7–1.0
- Scoring: Binary — correct=1.0, incorrect=0.0

### Task 2 — Compressed Detection (Medium)
- Description: Classify speech after codec compression degradation. Acoustic features are degraded, compression artifacts added.
- Difficulty: Medium
- Expected agent score: 0.5–0.9
- Scoring: Partial credit based on confidence calibration
  - correct + high confidence → 1.0
  - correct + low confidence → 0.6
  - wrong + low confidence → 0.2
  - wrong + high confidence → 0.0

### Task 3 — Adversarial Detection (Hard)
- Description: Synthetic audio specifically crafted to mimic real speech characteristics. Acoustic feature distributions overlap significantly with real speech, making clean threshold-based classification impossible.
- Difficulty: Hard
- Expected agent score: 0.3–0.6
- Scoring: Rewards correct classification AND penalizes overconfidence
  - correct + calibrated confidence → ~1.0
  - correct + overconfident → 0.5
  - wrong + appropriately uncertain → 0.15
  - wrong + overconfident → 0.0

---

## 🎁 Reward Function

The reward function provides partial, meaningful signals — not just binary win/lose.
```python
def grade(true_label, action, difficulty):
    correct = (action["label"] == true_label)
    confidence = action["confidence"]

    if difficulty == "easy":
        return 1.0 if correct else 0.0

    elif difficulty == "medium":
        if correct:
            return 0.6 + 0.4 * confidence
        else:
            return max(0.0, 0.2 - 0.3 * confidence)

    elif difficulty == "hard":
        if correct:
            base = 0.5
            calibration_bonus = 0.5 * (1 - abs(confidence - 0.7))
            return base + calibration_bonus
        else:
            return 0.15 if confidence < 0.4 else 0.0
```

### Why Confidence Calibration Matters
An agent that is wrong but uncertain is more useful than one that is wrong but confident. This reward design teaches agents to express appropriate uncertainty — critical for real-world fraud detection systems where a confident wrong answer causes more damage than an uncertain one.

---

## 🔌 OpenEnv API
```python
from environment.env import VoiceAuthenticityEnv

env = VoiceAuthenticityEnv(task_name="clean_detection")

obs = env.reset()
# obs.features      → 48-dim list
# obs.hint          → task context and key raw values
# obs.difficulty    → "easy"

action = {"label": 1, "confidence": 0.8, "reasoning": "low temporal instability"}
obs, reward, done, info = env.step(action)
# reward            → float in [0.0, 1.0]
# done              → True (one classification per episode)
# info["true_label"]→ ground truth

state = env.state()
```

---

## 📊 Baseline Scores

Agent: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router
Runs: 10 independent episodes per task
Metric: Average reward per episode

| Task | Difficulty | Avg Reward | Success Rate | Notes |
|------|-----------|------------|--------------|-------|
| clean_detection | Easy | 0.80 | 80% | Strong baseline on clean features |
| compressed_detection | Medium | 0.45 | 55% | Compression degrades acoustic signal |
| adversarial_detection | Hard | 0.68 | 65% | Calibration reward benefits uncertain agents |

Scores vary per run due to random sample selection from the 500-sample pool. Higher scores on harder tasks reflect the confidence calibration reward — agents that express appropriate uncertainty score better than overconfident wrong answers.

---

## ⚠️ Known Limitations and Failure Cases

- Synthetic voices with injected background noise may evade detection
- Real voices recorded under heavy studio compression may score lower than expected
- Borderline acoustic feature overlap exists between real and adversarially crafted samples — no clean threshold separates them
- Dataset of 500 samples is designed for evaluation structure and reward design, not production scale
- The feature pipeline supports arbitrary dataset scaling for enterprise deployment
- Results may vary across accents, languages, and recording conditions not represented in the training distribution

This environment is designed to be extended with real enterprise datasets. The evaluation structure, reward function, and feature pipeline are production-ready; the dataset is a research prototype.

---

## 🚀 Setup and Usage

### Requirements
Python 3.10+
Docker
HuggingFace account

### Local Setup
```bash
git clone https://huggingface.co/spaces/AksharaSharma/voice-authenticity-openenv
cd voice-authenticity-openenv

pip install -r requirements.txt

python scripts/download_data.py
python scripts/extract_features.py

cp .env.example .env
# Edit .env with your HF_TOKEN

python inference.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API token | required |
| `VOICE_TASK` | Task to run | `clean_detection` |

### Docker
```bash
docker build -t voice-authenticity .
docker run --env-file .env voice-authenticity
```

---

## 📁 Project Structure
voice-authenticity-openenv/
├── environment/
│   ├── init.py
│   ├── env.py              # step() / reset() / state()
│   ├── models.py           # Pydantic Observation/Action/Reward
│   ├── graders.py          # scoring logic per task
│   └── data/
│       ├── features.npy            # clean features (500 × 48)
│       ├── features_compressed.npy # codec-degraded features
│       ├── features_adversarial.npy# adversarially perturbed
│       ├── features_raw.npy        # unnormalized for hints
│       └── labels.npy              # ground truth labels
├── scripts/
│   ├── download_data.py    # fetch dataset from HuggingFace
│   └── extract_features.py # audio → feature vectors
├── server/
│   └── app.py              # OpenEnv HTTP server entry point
├── app.py                  # FastAPI server
├── inference.py            # baseline LLM agent
├── openenv.yaml            # OpenEnv spec
├── pyproject.toml          # package config
├── Dockerfile
├── requirements.txt
└── README.md

---

## 🔬 Technical Details

### Feature Extraction Pipeline
Audio (.wav / .flac)
↓ librosa (MFCCs, spectral features)
↓ parselmouth/Praat (jitter, shimmer, HNR)
↓ z-score normalization
↓ 48-dim float32 vector
→ stored as .npy arrays

### Compression Simulation (Task 2)
Codec compression is simulated by degrading MFCC standard deviations, reducing jitter and shimmer values, and adding spectral artifact signals to the compression artifact feature indices.

### Adversarial Simulation (Task 3)
Adversarial perturbation on synthetic samples introduces overlapping acoustic feature distributions with real speech. Jitter and shimmer values are shifted into the real speech range, making threshold-based classification unreliable and requiring genuine uncertainty quantification.

---

## 📋 Expected stdout Format
[START] task=clean_detection env=voice-authenticity model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"label":0,"confidence":0.75,"reasoning":"..."} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00

---

## 📜 License
MIT