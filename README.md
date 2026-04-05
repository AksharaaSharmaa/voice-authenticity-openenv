Here's your complete README:

---

```markdown
# 🎙️ Voice Authenticity Detection — OpenEnv Environment

An reinforcement learning environment for training and evaluating AI agents 
to detect synthetic (AI-generated) speech across real-world degradation 
conditions.

> Voice fraud is a growing crisis. This environment trains agents to detect 
> synthetic speech under clean, compressed, and adversarial conditions — 
> directly applicable to fraud detection, content moderation, and voice 
> authentication systems.

---

## 🌍 Real-World Motivation

AI-generated voices (ElevenLabs, Coqui, etc.) are increasingly used for:
- **Phone fraud** and social engineering attacks
- **Deepfake audio** in misinformation campaigns
- **Identity spoofing** in voice authentication systems

This environment provides a structured benchmark for training agents to 
detect synthetic speech under realistic degradation conditions that existing 
classifiers struggle with.

---

## 🏗️ Environment Overview

The environment serves **48-dimensional feature vectors** extracted from 
audio samples. Agents must classify each sample as real or synthetic, 
with a confidence score.

### Why Feature Vectors, Not Raw Audio?
- Fits within 2 vCPU / 8GB RAM constraints
- Feature extraction done offline — inference is fast
- Interpretable observations for LLM-based agents

### Dataset
- **Real speech**: 250 samples from 
  `garystafford/deepfake-audio-detection` (authentic human recordings)
- **Synthetic speech**: 250 samples (ElevenLabs, Hume AI, and other 
  TTS platforms)
- **Total**: 500 labeled samples across 3 task variants

---

## 📐 Observation Space

Each observation is a **48-dimensional float32 vector**:

| Index | Feature | Description |
|-------|---------|-------------|
| 0–19 | MFCC means | Timbre and spectral shape |
| 20–39 | MFCC std devs | Variation — synthetic voices are too stable |
| 40 | Zero crossing rate | Signal sign changes per frame |
| 41 | Spectral centroid | Brightness of the sound |
| 42 | Jitter | Frequency instability — real voices wobble slightly |
| 43 | Shimmer | Amplitude instability — real voices vary naturally |
| 44 | HNR | Harmonics-to-noise ratio — synthetic voices too clean |
| 45–47 | Compression artifacts | Spectral bandwidth, rolloff, RMS energy |

### Key Discriminators
```
Real speech:      jitter > 0.025, shimmer > 0.10, hnr < 12.0
Synthetic speech: jitter < 0.020, shimmer < 0.09, hnr > 12.0
```

### Observation Schema (Pydantic)
```python
class VoiceObservation(BaseModel):
    features: List[float]      # 48-dim feature vector (normalized)
    task_name: str             # current task
    step_number: int           # current step
    difficulty: str            # easy | medium | hard
    sample_id: int             # index into dataset
    hint: Optional[str]        # key raw values + task warning
```

---

## 🎯 Action Space

```python
class VoiceAction(BaseModel):
    label: int        # 0 = real, 1 = synthetic
    confidence: float # confidence in [0.0, 1.0]
    reasoning: str    # brief explanation
```

---

## 🏆 Tasks

### Task 1 — Clean Detection (Easy)
- **Description**: Classify real vs synthetic speech from clean, 
  unmodified audio features
- **Difficulty**: Easy
- **Expected agent score**: 0.7–1.0
- **Scoring**: Binary — correct=1.0, incorrect=0.0

### Task 2 — Compressed Detection (Medium)
- **Description**: Classify speech after codec compression degradation. 
  Jitter and shimmer are reduced, compression artifacts added.
- **Difficulty**: Medium
- **Expected agent score**: 0.5–0.9
- **Scoring**: Partial credit based on confidence calibration
  ```
  correct + high confidence → 1.0
  correct + low confidence  → 0.6
  wrong + low confidence    → 0.2
  wrong + high confidence   → 0.0
  ```

### Task 3 — Adversarial Detection (Hard)
- **Description**: Synthetic audio specifically crafted to mimic real 
  speech features. Jitter and shimmer are artificially elevated.
- **Difficulty**: Hard
- **Expected agent score**: 0.3–0.97
- **Scoring**: Rewards correct classification AND penalizes overconfidence
  ```
  correct + calibrated confidence (~0.7) → ~1.0
  correct + overconfident (>0.9)         → 0.5
  wrong + appropriately uncertain        → 0.15
  wrong + overconfident                  → 0.0
  ```

---

## 🎁 Reward Function

The reward function provides **partial, meaningful signals** — not just 
binary win/lose.

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
An agent that is **wrong but uncertain** is more useful than one that is 
**wrong but confident**. This reward design teaches agents to express 
appropriate uncertainty — critical for real-world fraud detection systems.

---

## 🔌 OpenEnv API

```python
from environment.env import VoiceAuthenticityEnv

env = VoiceAuthenticityEnv(task_name="clean_detection")

# Reset episode
obs = env.reset()
# obs.features      → 48-dim list
# obs.hint          → key raw values for interpretation
# obs.difficulty    → "easy"

# Take action
action = {"label": 1, "confidence": 0.8, "reasoning": "low jitter"}
obs, reward, done, info = env.step(action)
# reward            → float in [0.0, 1.0]
# done              → True (one classification per episode)
# info["true_label"]→ ground truth

# Get state
state = env.state()
```

---

## 📊 Baseline Scores

Scores from `Qwen/Qwen2.5-72B-Instruct` across multiple runs:

| Task | Difficulty | Avg Reward | Notes |
|------|-----------|------------|-------|
| clean_detection | Easy | ~0.80 | Strong baseline |
| compressed_detection | Medium | ~0.70 | Compression reduces confidence |
| adversarial_detection | Hard | ~0.75 | Calibration reward helps |

---

## 🚀 Setup & Usage

### Requirements
```
Python 3.10+
Docker
HuggingFace account
```

### Local Setup

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/voice-authenticity-openenv
cd voice-authenticity-openenv

# Install dependencies
pip install -r requirements.txt

# Download dataset and extract features
python scripts/download_data.py
python scripts/extract_features.py

# Set environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN

# Run baseline inference
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
# Build
docker build -t voice-authenticity .

# Run
docker run --env-file .env voice-authenticity
```

---

## 📁 Project Structure

```
voice-authenticity-openenv/
├── environment/
│   ├── __init__.py
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
├── inference.py            # baseline LLM agent
├── openenv.yaml            # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Details

### Feature Extraction Pipeline
```
Audio (.wav / .flac)
    ↓ librosa (MFCCs, spectral features)
    ↓ parselmouth/Praat (jitter, shimmer, HNR)
    ↓ z-score normalization
    ↓ 48-dim float32 vector
    → stored as .npy arrays
```

### Compression Simulation (Task 2)
Codec compression is simulated by:
- Degrading MFCC standard deviations (compression flattens variation)
- Reducing jitter and shimmer values
- Adding spectral artifact signals to indices 45–47

### Adversarial Simulation (Task 3)
Adversarial perturbation on synthetic samples:
- Artificially elevates jitter (+0.005 to +0.02)
- Artificially elevates shimmer (+0.01 to +0.05)
- Slightly reduces HNR to mimic real speech

---

## 📋 Expected stdout Format

```
[START] task=clean_detection env=voice-authenticity model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"label":0,"confidence":0.95,"reasoning":"..."} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
```

---

## 📜 License
MIT
```

---

## Save This

Create `README.md` in your project root and paste everything above into it.

Also create `.env.example` (safe to commit, no real token):

```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_huggingface_token_here
VOICE_TASK=clean_detection
```

---