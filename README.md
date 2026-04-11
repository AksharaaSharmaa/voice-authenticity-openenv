---
title: Voice Authenticity OpenEnv
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /docs
tags:
  - openenv
  - speech
  - fraud-detection
  - audio
---

# 🎙️ Voice Authenticity Detection : OpenEnv Environment

Voice fraud now costs the global economy over **$25 billion annually**, devastating banking, insurance, telecom, and government services. AI-generated voices from platforms like ElevenLabs, Coqui, and Bark can clone any voice in under 60 seconds : enabling real-time phone scams, identity theft, and social engineering at unprecedented scale. Existing benchmarks like ASVspoof and ADD fail under real-world conditions: they operate on static datasets with fixed train/test splits, evaluate single-shot classifiers with no agent interaction, ignore partial observability (real systems never see all features at once), and provide binary pass/fail scoring with no reward shaping. This environment fills that gap. It trains agents to **actively gather, analyze, and reason about acoustic evidence** under realistic degradation : codec compression, adversarial perturbation, streaming noise, and phone call simulation : through a genuine multi-step decision process with 5 distinct actions, 6-component grading, and step-level reward shaping that teaches calibrated, risk-aware classification.

---

## 🌍 Real-World Motivation

AI-generated voices are increasingly weaponized for:

- **Phone fraud & social engineering** : real-time voice cloning during live calls
- **Deepfake audio in misinformation** : fabricated audio of public figures
- **Identity spoofing** : bypassing voice biometric authentication systems
- **Financial fraud** : CEO voice cloning for unauthorized wire transfers
- **Insurance scams** : fabricated recorded statements

This environment provides a structured benchmark for training agents to detect synthetic speech under conditions that static classifiers and existing benchmarks cannot handle.

---

## 🏗️ Environment Overview

The environment serves 48-dimensional feature vectors extracted from audio samples. Unlike standard classification benchmarks, agents **start with NO features visible** and must actively query the environment through a 5-action protocol to gather evidence before making a final classification.

This creates genuine **sequential decision-making under partial observability**, requiring agents to:
- Choose which information to request and in what order
- Synthesize heterogeneous evidence sources
- Express calibrated confidence reflecting genuine uncertainty
- Follow logical investigation trajectories

---

## 🧠 Agent Interaction Model (5-Action Multi-Step)

The agent interacts through **5 distinct actions**, each returning genuinely different observation content:

| Action | Returns | Purpose |
|--------|---------|---------|
| `request_temporal_features` | Jitter, shimmer, HNR (raw + normalized) | Vocal cord irregularity markers |
| `request_spectral_features` | 20 MFCC means, 20 MFCC stds, ZCR, spectral centroid | Timbre and spectral shape |
| `request_comparison` | Cosine similarity + euclidean distance to real/fake centroids | Statistical comparison to known references |
| `analyze_evidence` | Structured synthesis of all gathered evidence with signal tally | Evidence integration and confidence calibration |
| `final_classify` | Submits label (0=real, 1=synthetic) + confidence + reasoning | Terminal action : triggers 6-component grading |

### Key Design Properties

- **Partial observability** : features are zeroed until explicitly requested
- **Action-dependent observations** : each action reveals genuinely different data
- **Flexible ordering** : agent chooses its own investigation strategy
- **Soft-gated streaming** : streaming task adds step-dependent noise (noisier early, cleaner late)
- **Step-level rewards** : shaping signals throughout the episode, not just at the end

Episodes consist of **up to 6 steps** (5 investigation actions + buffer), not a single prediction.

---

## ⚙️ Why Feature Vectors Instead of Raw Audio?

- Fits within 2 vCPU / 8GB RAM constraints
- Feature extraction is performed offline for fast inference
- Enables **LLM-native reasoning over interpretable acoustic characteristics** : not possible with raw waveforms under current infrastructure constraints
- Avoids heavy signal processing during evaluation

---

## 📊 Dataset

- Real speech: 250 samples from `garystafford/deepfake-audio-detection` (authentic human recordings)
- Synthetic speech: 250 samples (ElevenLabs, Hume AI, and other TTS platforms)
- Total: 500 labeled samples across 5 task variants

The dataset is designed for **evaluation structure and reward learning**, not scale. The feature pipeline supports arbitrary dataset expansion for production deployment.

---

## 📐 Observation Space

Each observation contains:

```python
class VoiceObservation(BaseModel):
    features: List[float]                          # 48-dim (zeroed until revealed)
    task_name: str                                 # current task
    step_number: int                               # current step in episode
    difficulty: str                                # easy|medium|medium_hard|hard|extreme
    sample_id: int                                 # index into dataset
    hint: Optional[str]                            # context and guidance
    visible_features: Dict[str, Any]               # features revealed so far
    evidence_summary: Optional[str]                # from analyze_evidence
    comparison_result: Optional[Dict[str, float]]  # from request_comparison
    available_actions: List[str]                    # valid actions this step
    actions_taken: List[str]                        # action history
```

### 48-Dimensional Feature Vector

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

- **Jitter**: measures cycle-to-cycle frequency instability : real voices show natural irregularity, synthetic voices are too stable
- **Shimmer**: tracks amplitude variation between consecutive glottal pulses : real speech has organic variation
- **HNR**: quantifies harmonic-to-noise ratio : synthetic voices are typically "too clean"

---

## 🎯 Action Space

```python
class VoiceAction(BaseModel):
    action_type: str   # one of the 5 actions
    label: int         # 0=real, 1=synthetic (for final_classify)
    confidence: float  # [0.05, 0.95] (for final_classify)
    reasoning: str     # explanation (for final_classify)
```

---

## 🏆 Tasks (5 Total)

### Task 1 : Clean Detection (Easy)

- **Description**: Classify real vs synthetic speech from clean, unmodified audio features
- **Difficulty**: Easy
- **Expected agent score**: 0.7–0.95

### Task 2 : Compressed Detection (Medium)

- **Description**: Classify speech after codec compression degradation. MFCC stds are flattened, jitter/shimmer are suppressed, spectral artifacts are added.
- **Difficulty**: Medium
- **Expected agent score**: 0.4–0.7

### Task 3 : Adversarial Detection (Hard)

- **Description**: Synthetic audio engineered to mimic real speech characteristics. Feature distributions overlap significantly with real speech. 8% label noise simulates real-world annotation ambiguity.
- **Difficulty**: Hard
- **Expected agent score**: 0.3–0.6

### Task 4 : Streaming Detection (Medium-Hard)

- **Description**: Multi-step streaming scenario where features arrive with step-dependent noise. Earlier requests return noisier data; later requests return cleaner data. Agents are rewarded for intelligent sequencing without being forced into a fixed order (soft-gating).
- **Difficulty**: Medium-Hard
- **Expected agent score**: 0.3–0.6

### Task 5 : Phone Call Detection (Extreme)

- **Description**: Simulates worst-case real-world conditions: heavy narrowband codec compression (300-3400Hz telephony simulation), additive background noise across all frequency bands, severe HNR degradation, MFCC high-frequency rolloff, and RMS energy fluctuation from packet loss. Designed to be near the limit of detectability.
- **Difficulty**: Extreme
- **Expected agent score**: 0.2–0.5

---

## 🏅 Grading System (6 Components)

Each episode is scored across 6 components with difficulty-weighted contributions:

| Component | What It Measures | Easy | Medium | Hard | Extreme |
|-----------|-----------------|------|--------|------|---------|
| **Correctness** | Label matches ground truth | 0.40 | 0.30 | 0.25 | 0.20 |
| **Confidence Calibration** | Penalizes overconfidence, rewards calibrated uncertainty | 0.15 | 0.20 | 0.25 | 0.25 |
| **Trajectory Quality** | Did agent gather → analyze → classify? | 0.10 | 0.15 | 0.18 | 0.20 |
| **Feature Utilization** | Did agent request temporal AND spectral features? | 0.15 | 0.15 | 0.12 | 0.15 |
| **Reasoning Consistency** | Does reasoning text match chosen label? | 0.10 | 0.10 | 0.10 | 0.10 |
| **Action Ordering** | Logical sequence: gather → analyze → classify | 0.10 | 0.10 | 0.10 | 0.10 |

### Why This Matters

On easy tasks, correctness dominates. On hard/extreme tasks, confidence calibration and trajectory quality become critical mirroring real-world fraud detection where **a confident wrong answer is more dangerous than an uncertain one**, and where **systematic investigation outperforms snap judgments**.

---

## 🎁 Step-Level Rewards

The environment provides shaping signals at every step, not just on final classification:

| Condition | Reward |
|-----------|--------|
| First action is a feature request | +0.05 |
| Requested both temporal AND spectral features | +0.05 |
| Used `analyze_evidence` before `final_classify` | +0.05 |
| Jumped straight to `final_classify` without gathering | -0.10 |
| Repeated the same action consecutively | -0.05 |
| Reasoning contradicts chosen label | -0.10 |

These intermediate rewards teach agents **investigation behavior** rather than pure classification.

---

## 🔌 OpenEnv API

```python
from environment.env import VoiceAuthenticityEnv

env = VoiceAuthenticityEnv(task_name="clean_detection")

# Reset : no features visible yet
obs = env.reset()
# obs.features           → [0.05, 0.05, ..., 0.05] (zeroed)
# obs.available_actions  → ["request_temporal_features", ...]

# Step 1 : request temporal features
action = {"action_type": "request_temporal_features"}
obs, reward, done, info = env.step(action)
# obs.visible_features["temporal"]["jitter"] → 0.032451
# reward → 0.05 (shaping: first action is gathering)

# Step 2 : request spectral features
action = {"action_type": "request_spectral_features"}
obs, reward, done, info = env.step(action)
# obs.visible_features["spectral"]["mfcc_means"] → [20 values]
# reward → 0.05 (shaping: multi-feature-type bonus)

# Step 3 : compare to reference centroids
action = {"action_type": "request_comparison"}
obs, reward, done, info = env.step(action)
# obs.comparison_result["cosine_similarity_to_real"] → 0.8742
# obs.comparison_result["closer_to"] → "real"

# Step 4 : analyze all evidence
action = {"action_type": "analyze_evidence"}
obs, reward, done, info = env.step(action)
# obs.evidence_summary → "Evidence analysis (3 sources): ..."

# Step 5 : final classification
action = {
    "action_type": "final_classify",
    "label": 0,
    "confidence": 0.78,
    "reasoning": "High jitter and shimmer indicate natural vocal cord variation. HNR is low, consistent with real speech. Comparison confirms closer to real centroid."
}
obs, reward, done, info = env.step(action)
# reward → 0.87 (6-component graded score)
# done → True
# info["grader_breakdown"] → {correctness: 0.95, calibration: 0.84, ...}

state = env.state()
```

---

## 📊 Baseline Scores

Agent: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router
Protocol: 5-action (temporal → spectral → comparison → analyze → classify)
Runs: 10 independent episodes per task

| Task | Difficulty | Avg Reward | Success Rate | Notes |
|------|-----------|------------|--------------|-------|
| clean_detection | Easy | 0.80 | 80% | Strong baseline on clean features |
| compressed_detection | Medium | 0.45 | 55% | Compression degrades acoustic signal |
| adversarial_detection | Hard | 0.50 | 50% | Overlapping distributions challenge models |
| streaming_detection | Medium-Hard | 0.40 | 45% | Soft-gated noise reduces early accuracy |
| phonecall_detection | Extreme | 0.30 | 35% | Near detection limit under phone conditions |

Scores vary per run due to random sample selection. Higher rewards on harder tasks reflect confidence calibration : agents that express appropriate uncertainty score better than overconfident wrong answers.

---

## ⚠️ Known Limitations and Failure Cases

- Synthetic voices with injected background noise may evade temporal feature detection
- Real voices under heavy studio compression can mimic synthetic spectral profiles
- Borderline acoustic feature overlap exists between real and adversarially crafted samples : no clean threshold separates them
- Phone call simulation pushes detection to near-chance performance, reflecting genuine real-world difficulty
- Streaming task noise is step-dependent : agents that don't re-request features may work from degraded data
- Dataset of 500 samples is designed for evaluation structure and reward design, not production scale
- Results may vary across accents, languages, and recording conditions not represented in the data

This environment is designed to be extended with real enterprise datasets. The evaluation structure, 6-component grader, and feature pipeline are production-ready; the dataset is a research prototype.

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

# Terminal 1 : start the environment server
python app.py

# Terminal 2 : run baseline inference (5-action protocol, all 5 tasks)
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
│   ├── env.py              # 5-action step/reset/state with partial observability
│   ├── models.py           # Pydantic Observation/Action/Reward models
│   ├── graders.py          # 6-component scoring with difficulty weights
│   └── data/
│       ├── features.npy            # clean features (500 × 48)
│       ├── features_compressed.npy # codec-degraded features
│       ├── features_adversarial.npy# adversarially perturbed features
│       ├── features_streaming.npy  # streaming degraded features
│       ├── features_phonecall.npy  # phone call degraded features
│       ├── features_raw.npy        # unnormalized values
│       ├── labels.npy              # ground truth labels
│       ├── labels_compressed.npy
│       ├── labels_adversarial.npy
│       ├── labels_streaming.npy
│       └── labels_phonecall.npy
├── scripts/
│   ├── download_data.py    # fetch dataset from HuggingFace
│   └── extract_features.py # audio → feature vectors (5 tasks)
├── server/
│   └── app.py              # OpenEnv HTTP server entry point
├── app.py                  # FastAPI server (root)
├── inference.py            # baseline LLM agent (5-action protocol)
├── openenv.yaml            # OpenEnv spec (5 tasks)
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
    → stored as .npy arrays (5 variants)
```

### Compression Simulation (Task 2)
Codec compression is simulated by degrading MFCC standard deviations, reducing jitter and shimmer values, and adding spectral artifact signals : replicating the acoustic degradation introduced by MP3/codec pipelines.

### Adversarial Simulation (Task 3)
Adversarial perturbation shifts synthetic sample features into the real speech distribution range, and real sample features toward the synthetic range. Controlled label noise (8%) simulates real-world annotation ambiguity. No clean threshold separates the classes.

### Streaming Simulation (Task 4)
Features undergo two layers of degradation: a static perturbation (partial MFCC decode, mild temporal noise) baked into the data files, and a dynamic soft-gated noise applied at runtime that reduces as the agent takes more steps. Early requests return noisier data; later requests return cleaner data : rewarding intelligent sequencing without forcing a fixed order.

### Phone Call Simulation (Task 5)
The most aggressive degradation: narrowband codec compression zeros out high-order MFCCs, flattens MFCC temporal variation, injects broadband Gaussian noise, severely degrades HNR, and adds RMS energy fluctuation simulating packet loss. Designed to be near the limit of what's detectable.

---

## 📋 Expected stdout Format
```
[START] task=clean_detection env=voice-authenticity model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type": "request_temporal_features"} reward=0.05 done=false error=null
[STEP] step=2 action={"action_type": "request_spectral_features"} reward=0.05 done=false error=null
[STEP] step=3 action={"action_type": "request_comparison"} reward=0.05 done=false error=null
[STEP] step=4 action={"action_type": "analyze_evidence"} reward=0.05 done=false error=null
[STEP] step=5 action={"action_type": "final_classify", "label": 0, "confidence": 0.78, "reasoning": "..."} reward=0.87 done=true error=null
[END] success=true steps=5 score=0.870 rewards=0.05,0.05,0.05,0.05,0.87
```

---

## 📜 License

MIT