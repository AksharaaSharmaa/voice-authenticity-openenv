---
title: Voice Authenticity OpenEnv
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🎙️ Voice Authenticity Detection — OpenEnv Environment

An advanced reinforcement learning environment for training and evaluating AI agents to detect synthetic (AI-generated) speech across real-world degradation and adversarial conditions.

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

The environment serves 48-dimensional feature vectors extracted from audio samples. Agents must classify each sample as real or synthetic, while expressing calibrated confidence that reflects genuine uncertainty.

Unlike standard classification benchmarks, this environment introduces **sequential decision-making under partial observability**, requiring agents to actively query and interpret information before making a final decision.

---

## 🧠 Agent Interaction Model (Multi-Step)

This environment operates as a **two-phase decision process**, transforming it from a static classification task into a true agentic system:

**Phase 1 — Analyze**
The agent inspects the observation and requests focus on specific acoustic features (jitter, shimmer, spectral properties, etc.).

**Phase 2 — Decide**
The agent submits a final classification (`real` or `synthetic`) along with a confidence score and reasoning.

This structure introduces:
- Partial observability
- Action-dependent state transitions
- Planning and tool-use behavior

Episodes consist of a **two-step interaction (analysis → decision)** rather than a single-step prediction.

---

## ⚙️ Why Feature Vectors Instead of Raw Audio?

- Fits within 2 vCPU / 8GB RAM constraints
- Feature extraction is performed offline for fast inference
- Enables **LLM-native reasoning over interpretable audio characteristics**, which is not possible with raw waveforms under current infrastructure constraints
- Avoids heavy signal processing during evaluation

---

## 📊 Dataset

- Real speech: 250 samples from `garystafford/deepfake-audio-detection` (authentic human recordings)
- Synthetic speech: 250 samples (ElevenLabs, Hume AI, and other TTS platforms)
- Total: 500 labeled samples across 3 task variants

The dataset is designed for **evaluation structure and reward learning**, not scale. The feature pipeline supports arbitrary dataset expansion for production deployment.

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

These acoustic properties capture fundamental differences between human and synthetic vocal production without exposing trivial decision rules:

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
- Expected agent score: 0.4–0.7
- Scoring: Partial credit based on confidence calibration
  - correct + high confidence → 1.0
  - correct + low confidence → 0.6
  - wrong + low confidence → 0.2
  - wrong + high confidence → 0.0

### Task 3 — Adversarial Detection (Hard)

- Description: Synthetic audio specifically engineered to mimic real speech characteristics. Feature distributions overlap significantly with real speech, making threshold-based classification unreliable.
- Difficulty: Hard
- Expected agent score: 0.3–0.6
- Additional realism: distribution overlap between classes, controlled ambiguity, label noise to simulate real-world inconsistencies
- Scoring: Rewards correct classification AND penalizes overconfidence
  - correct + calibrated confidence (~0.7) → ~1.0
  - correct + overconfident (>0.9) → 0.5
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

In real-world fraud detection systems, a **confident wrong prediction is more dangerous than an uncertain one**. This environment explicitly rewards:

- Calibrated uncertainty
- Risk-aware decision-making
- Avoidance of overconfident errors

---

## 🔌 OpenEnv API
```python
from environment.env import VoiceAuthenticityEnv

env = VoiceAuthenticityEnv(task_name="clean_detection")

obs = env.reset()
# obs.features      → 48-dim list
# obs.hint          → phase instructions and key raw values
# obs.difficulty    → "easy"

# Phase 1 — analysis request
action = {"focus": ["jitter", "shimmer", "hnr"], "label": 0, "confidence": 0.5, "reasoning": "requesting analysis"}
obs, reward, done, info = env.step(action)
# info["phase"] → "decide"

# Phase 2 — final decision
action = {"label": 1, "confidence": 0.75, "reasoning": "low temporal instability"}
obs, reward, done, info = env.step(action)
# reward            → float in [0.0, 1.0]
# done              → True
# info["true_label"]→ ground truth

state = env.state()
```

---

## 📊 Baseline Scores

Agent: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router
Runs: 10 independent episodes per task
Metric: Average reward per episode (decision phase only)

| Task | Difficulty | Avg Reward | Success Rate | Notes |
|------|-----------|------------|--------------|-------|
| clean_detection | Easy | 0.80 | 80% | Strong baseline on clean features |
| compressed_detection | Medium | 0.45 | 55% | Compression degrades acoustic signal |
| adversarial_detection | Hard | 0.50 | 50% | Overlapping distributions challenge frontier models |

Scores vary per run due to random sample selection. Higher rewards on harder tasks reflect the confidence calibration reward — agents that express appropriate uncertainty score better than overconfident wrong answers.

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
```
Python 3.10+
Docker
HuggingFace account
```

### Local Setup
```bash
git clone https://huggingface.co/spaces/AksharaSharma/voice-authenticity-openenv
cd voice-authenticity-openenv

pip install -r requirements.txt

python scripts/download_data.py
python scripts/extract_features.py

cp .env.example .env
# Edit .env with your HF_TOKEN

# Terminal 1 — start the environment server
python app.py

# Terminal 2 — run baseline inference
python inference.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API token | required |
| `VOICE_TASK` | Task to run | `clean_detection` |
| `ENV_SERVER_URL` | Environment server URL | `http://localhost:7860` |

### Docker
```bash
docker build -t voice-authenticity .
docker run --env-file .env voice-authenticity
```

---

## 📁 Project Structure
```
voice-authenticity-openenv/
├── environment/
│   ├── __init__.py
│   ├── env.py              # step() / reset() / state() with 2-phase loop
│   ├── models.py           # Pydantic Observation/Action/Reward models
│   ├── graders.py          # scoring logic per task
│   └── data/
│       ├── features.npy            # clean features (500 × 48)
│       ├── features_compressed.npy # codec-degraded features
│       ├── features_adversarial.npy# adversarially perturbed features
│       ├── features_raw.npy        # unnormalized values for hints
│       ├── labels.npy              # ground truth labels
│       ├── labels_compressed.npy
│       └── labels_adversarial.npy
├── scripts/
│   ├── download_data.py    # fetch dataset from HuggingFace
│   └── extract_features.py # audio → feature vectors
├── server/
│   └── app.py              # OpenEnv HTTP server entry point
├── app.py                  # FastAPI server (root)
├── inference.py            # baseline LLM agent
├── openenv.yaml            # OpenEnv spec
├── pyproject.toml          # package config
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Pipeline

### Feature Extraction
```
Audio (.wav / .flac)
    ↓ librosa → MFCC + spectral features
    ↓ parselmouth/Praat → jitter, shimmer, HNR
    ↓ z-score normalization
    ↓ 48-dim float32 vector
    → stored as .npy arrays
```

### Compression Simulation (Task 2)
Codec compression is simulated by degrading MFCC standard deviations, reducing jitter and shimmer values, and adding spectral artifact signals — replicating the acoustic degradation introduced by MP3/codec pipelines.

### Adversarial Simulation (Task 3)
Adversarial perturbation shifts synthetic sample features into the real speech distribution range, and real sample features toward the synthetic range. Controlled label noise (8%) is introduced to simulate real-world annotation ambiguity. No clean threshold-based separation exists between classes.

---

## 📋 Expected stdout Format
```
[START] task=clean_detection env=voice-authenticity model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"focus":["jitter","shimmer","hnr"],"label":0,"confidence":0.5,"reasoning":"Requesting focused analysis"} reward=0.00 done=false error=null
[STEP] step=2 action={"label":0,"confidence":0.75,"reasoning":"..."} reward=1.00 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.00,1.00
```

---

## 📜 License

MIT