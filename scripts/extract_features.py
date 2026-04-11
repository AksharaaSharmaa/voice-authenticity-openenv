import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import os
import warnings
warnings.filterwarnings("ignore")

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"
OUTPUT_DIR = "environment/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features(file_path):
    """
    Extract 48-dim feature vector from audio file.
    Returns None if file fails.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)

        if len(y) < 1600:  # skip clips shorter than 0.1s
            return None

        # ── MFCC (40 features) ──────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = mfcc.mean(axis=1)   # 20 values
        mfcc_std  = mfcc.std(axis=1)    # 20 values

        # ── Spectral features (2 features) ──────────────────
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spec_centroid = librosa.feature.spectral_centroid(
                            y=y, sr=sr).mean()

        # ── Voice authenticity features (3 features) ────────
        # These are the KEY discriminators between real and fake
        try:
            snd = parselmouth.Sound(file_path)
            pp  = call(snd, "To PointProcess (periodic, cc)", 75, 500)

            jitter = call(
                pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            )
            shimmer = call(
                [snd, pp], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )
            harmonicity = call(
                snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
            )
            hnr = call(harmonicity, "Get mean", 0, 0)

            # Replace NaN/inf with 0
            jitter  = float(jitter)  if np.isfinite(jitter)  else 0.0
            shimmer = float(shimmer) if np.isfinite(shimmer) else 0.0
            hnr     = float(hnr)     if np.isfinite(hnr)     else 0.0

        except Exception:
            jitter, shimmer, hnr = 0.0, 0.0, 0.0

        # ── Compression artifact features (3 features) ──────
        # Simulates codec degradation for task 2
        spec_bandwidth = librosa.feature.spectral_bandwidth(
                             y=y, sr=sr).mean()
        spec_rolloff   = librosa.feature.spectral_rolloff(
                             y=y, sr=sr).mean()
        rms            = librosa.feature.rms(y=y).mean()

        # ── Assemble final 48-dim vector ─────────────────────
        features = np.concatenate([
            mfcc_mean,                          # 0-19
            mfcc_std,                           # 20-39
            [zcr, spec_centroid],               # 40-41
            [jitter, shimmer, hnr],             # 42-44
            [spec_bandwidth, spec_rolloff, rms] # 45-47
        ])

        return features.astype(np.float32)

    except Exception as e:
        print(f"  ERROR on {file_path}: {e}")
        return None


def process_directory(directory, label, desc):
    files = [
        f for f in os.listdir(directory)
        if f.endswith((".wav", ".flac", ".mp3"))
    ]
    print(f"\nProcessing {desc}: {len(files)} files found")

    features_list = []
    labels_list   = []
    failed         = 0

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        feat = extract_features(path)

        if feat is not None:
            features_list.append(feat)
            labels_list.append(label)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} done...")
        else:
            failed += 1

    print(f"  Success: {len(features_list)}, Failed: {failed}")
    return features_list, labels_list


def add_compression_artifacts(features, strength=0.3):
    """Simulate codec compression degradation."""
    degraded = features.copy()

    degraded[20:40] *= (1 - strength * np.random.uniform(0.5, 1.0, 20))
    degraded[42] *= (1 - strength * np.random.uniform(0.3, 0.7))
    degraded[43] *= (1 - strength * np.random.uniform(0.3, 0.7))
    degraded[44] *= (1 + strength * np.random.uniform(0.1, 0.4))
    degraded[45] *= (1 + strength * np.random.uniform(0.3, 0.8))
    degraded[46] *= (1 - strength * np.random.uniform(0.2, 0.6))
    degraded[47] += strength * np.random.uniform(0.1, 0.4)

    return degraded


def add_adversarial_perturbation(features, label):
    """
    True adversarial: create overlapping distributions.
    Fake audio shifted INTO real speech range.
    Real audio shifted TOWARD synthetic range.
    No clean threshold can separate them.
    """
    perturbed = features.copy()

    if label == 1:  # fake → make it look real
        # Push jitter into real range
        perturbed[42] += np.random.uniform(0.010, 0.025)
        # Push shimmer into real range
        perturbed[43] += np.random.uniform(0.020, 0.060)
        # Lower HNR toward real range
        perturbed[44] -= np.random.uniform(2.0, 5.0)
        # Add slight MFCC variation
        perturbed[20:30] += np.random.normal(0, 0.3, 10)

    elif label == 0:  # real → push toward synthetic range
        # Suppress jitter slightly
        perturbed[42] *= np.random.uniform(0.6, 0.85)
        # Suppress shimmer slightly
        perturbed[43] *= np.random.uniform(0.6, 0.85)
        # Raise HNR slightly
        perturbed[44] += np.random.uniform(0.5, 2.0)

    # Add 8% label noise — some samples are deliberately mislabeled
    # to simulate real-world distribution ambiguity
    if np.random.random() < 0.08:
        perturbed += np.random.normal(0, 0.5, len(perturbed))

    return perturbed


def add_streaming_degradation(features, label):
    """Simulate streaming/partial decode degradation.

    Models real-time audio streaming where features are partially decoded:
    - MFCC values slightly degraded (simulating partial frame decode)
    - Temporal features intact but with mild additive noise
    - High-frequency spectral features mildly rolled off
    - Overall mild Gaussian noise across all dims

    This is the base perturbation for Task 4 (streaming_detection).
    The environment also applies step-dependent soft-gated noise at runtime.
    """
    degraded = features.copy()

    # Partial MFCC decode: higher-order coefficients more degraded
    for i in range(20):
        degradation = 0.02 * (i / 20)  # more degradation on higher coeffs
        degraded[i] += np.random.normal(0, degradation + 0.01)
    for i in range(20, 40):
        degradation = 0.03 * ((i - 20) / 20)
        degraded[i] *= (1 - degradation * np.random.uniform(0.3, 0.8))

    # Mild noise on temporal features
    degraded[42] += np.random.normal(0, 0.003)   # jitter noise
    degraded[43] += np.random.normal(0, 0.008)   # shimmer noise
    degraded[44] += np.random.normal(0, 0.5)     # HNR noise

    # Mild spectral rolloff
    degraded[41] *= np.random.uniform(0.92, 0.98)   # spectral centroid
    degraded[45] *= np.random.uniform(0.90, 0.97)   # spectral bandwidth
    degraded[46] *= np.random.uniform(0.88, 0.96)   # spectral rolloff

    # Global mild noise
    degraded += np.random.normal(0, 0.015, len(degraded))

    return degraded


def add_phonecall_degradation(features, label):
    """Simulate phone call conditions: heavy codec + background noise.

    Models the worst-case real-world scenario:
    - Aggressive codec compression (narrowband telephony)
    - Background noise injection across all bands
    - Severe HNR degradation (noisy channel)
    - MFCC high-frequency rolloff (narrowband 300-3400Hz telephony)
    - RMS energy fluctuation (network jitter/packet loss)
    - Jitter/shimmer partially masked by channel noise

    This is the hardest task — designed to be near the limit of detectability.
    """
    degraded = features.copy()

    # ── Heavy codec compression (narrowband telephony simulation) ───
    # MFCC means: zero out high-order coefficients (narrowband kills them)
    for i in range(12, 20):
        degraded[i] *= np.random.uniform(0.1, 0.4)  # severe suppression
    # MFCC stds: flatten temporal variation (codec smoothing)
    degraded[20:40] *= np.random.uniform(0.3, 0.6, 20)

    # ── Background noise injection ──────────────────────────────────
    noise_strength = np.random.uniform(0.15, 0.35)
    degraded += np.random.normal(0, noise_strength, len(degraded))

    # ── Severe HNR degradation (noisy channel) ─────────────────────
    degraded[44] -= np.random.uniform(3.0, 7.0)  # massive HNR drop

    # ── Jitter/shimmer partially masked by channel noise ───────────
    degraded[42] += np.random.normal(0, 0.015)  # large jitter noise
    degraded[43] += np.random.normal(0, 0.03)   # large shimmer noise

    # ── Spectral degradation (narrowband rolloff) ──────────────────
    degraded[41] *= np.random.uniform(0.5, 0.75)   # centroid drops
    degraded[45] *= np.random.uniform(0.4, 0.65)   # bandwidth severely narrows
    degraded[46] *= np.random.uniform(0.3, 0.55)   # rolloff drastically drops

    # ── RMS energy fluctuation (packet loss / AGC) ──────────────────
    degraded[47] *= np.random.uniform(0.5, 1.5)

    # ── ZCR noise (transmission artifacts) ──────────────────────────
    degraded[40] += np.random.normal(0, 0.02)

    return degraded


def main():
    print("=" * 60)
    print("Feature Extraction Pipeline (5 Tasks)")
    print("=" * 60)

    real_feat, real_labels = process_directory(
        REAL_DIR, label=0, desc="REAL audio"
    )

    fake_feat, fake_labels = process_directory(
        FAKE_DIR, label=1, desc="FAKE audio"
    )

    all_features = np.array(real_feat + fake_feat, dtype=np.float32)
    all_labels   = np.array(real_labels + fake_labels, dtype=np.int32)

    idx = np.random.permutation(len(all_labels))
    all_features = all_features[idx]
    all_labels   = all_labels[idx]

    mean = all_features.mean(axis=0)
    std  = all_features.std(axis=0) + 1e-8
    all_features_norm = (all_features - mean) / std

    np.save(f"{OUTPUT_DIR}/features.npy", all_features_norm)

    # Save raw unnormalized features for env to use
    np.save(f"{OUTPUT_DIR}/features_raw.npy", all_features)

    np.save(f"{OUTPUT_DIR}/labels.npy", all_labels)
    np.save(f"{OUTPUT_DIR}/mean.npy", mean)
    np.save(f"{OUTPUT_DIR}/std.npy", std)

    print(f"\nTask 1 (clean): {len(all_labels)} samples saved")

    # ── TASK 2: Compressed features ─────────────────────────
    raw_combined = real_feat + fake_feat

    compressed_features = np.array([
        add_compression_artifacts(f, strength=0.3)
        for f in raw_combined
    ], dtype=np.float32)

    compressed_features = compressed_features[idx]
    compressed_norm = (compressed_features - mean) / std

    np.save(f"{OUTPUT_DIR}/features_compressed.npy", compressed_norm)
    np.save(f"{OUTPUT_DIR}/labels_compressed.npy", all_labels)

    print(f"Task 2 (compressed): {len(all_labels)} samples saved")

    # ── TASK 3: Adversarial features ────────────────────────
    raw_labels_combined = real_labels + fake_labels

    adversarial_features = np.array([
        add_adversarial_perturbation(f, l)
        for f, l in zip(raw_combined, raw_labels_combined)
    ], dtype=np.float32)

    adversarial_features = adversarial_features[idx]
    adversarial_norm = (adversarial_features - mean) / std

    np.save(f"{OUTPUT_DIR}/features_adversarial.npy", adversarial_norm)
    np.save(f"{OUTPUT_DIR}/labels_adversarial.npy", all_labels)

    print(f"Task 3 (adversarial): {len(all_labels)} samples saved")

    # ── TASK 4: Streaming features ──────────────────────────
    streaming_features = np.array([
        add_streaming_degradation(f, l)
        for f, l in zip(raw_combined, raw_labels_combined)
    ], dtype=np.float32)

    streaming_features = streaming_features[idx]
    streaming_norm = (streaming_features - mean) / std

    np.save(f"{OUTPUT_DIR}/features_streaming.npy", streaming_norm)
    np.save(f"{OUTPUT_DIR}/labels_streaming.npy", all_labels)

    print(f"Task 4 (streaming): {len(all_labels)} samples saved")

    # ── TASK 5: Phone call features ─────────────────────────
    phonecall_features = np.array([
        add_phonecall_degradation(f, l)
        for f, l in zip(raw_combined, raw_labels_combined)
    ], dtype=np.float32)

    phonecall_features = phonecall_features[idx]
    phonecall_norm = (phonecall_features - mean) / std

    np.save(f"{OUTPUT_DIR}/features_phonecall.npy", phonecall_norm)
    np.save(f"{OUTPUT_DIR}/labels_phonecall.npy", all_labels)

    print(f"Task 5 (phonecall): {len(all_labels)} samples saved")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"Total samples : {len(all_labels)}")
    print(f"Real samples  : {all_labels.tolist().count(0)}")
    print(f"Fake samples  : {all_labels.tolist().count(1)}")
    print(f"Feature shape : {all_features_norm.shape}")
    print(f"{'='*60}")

    print("\nSanity check — jitter/shimmer/HNR comparison:")
    for i in range(min(2, len(all_labels))):
        raw_i = np.array(raw_combined)[idx][i]
        label_str = "REAL" if all_labels[i] == 0 else "FAKE"
        print(f"\n  [{label_str}]")
        print(f"  Clean       → jitter={raw_i[42]:.4f} shimmer={raw_i[43]:.4f} hnr={raw_i[44]:.4f}")
        print(f"  Compressed  → jitter={compressed_features[i][42]:.4f} shimmer={compressed_features[i][43]:.4f} hnr={compressed_features[i][44]:.4f}")
        print(f"  Adversarial → jitter={adversarial_features[i][42]:.4f} shimmer={adversarial_features[i][43]:.4f} hnr={adversarial_features[i][44]:.4f}")
        print(f"  Streaming   → jitter={streaming_features[i][42]:.4f} shimmer={streaming_features[i][43]:.4f} hnr={streaming_features[i][44]:.4f}")
        print(f"  PhoneCall   → jitter={phonecall_features[i][42]:.4f} shimmer={phonecall_features[i][43]:.4f} hnr={phonecall_features[i][44]:.4f}")


if __name__ == "__main__":
    main()