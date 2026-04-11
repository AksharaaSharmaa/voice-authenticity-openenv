"""
Stress-test the DEPLOYED HF Space to find any score that is exactly 0.0 or 1.0.
Tests ALL 5 tasks with multiple agent behaviors.
"""
import requests
import json

BASE = "https://aksharasharma-voice-authenticity-openenv.hf.space"

def reset(task, seed=7):
    r = requests.post(f"{BASE}/reset", json={"task_name": task, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def step(action, task):
    payload = {
        "action_type": action.get("action_type", "final_classify"),
        "label": action.get("label", 0),
        "confidence": action.get("confidence", 0.5),
        "reasoning": action.get("reasoning", ""),
        "task_name": task,
    }
    r = requests.post(f"{BASE}/step", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def check_reward(reward, context):
    if reward <= 0.0 or reward >= 1.0:
        print(f"  *** VIOLATION: reward={reward} at {context}")
        return False
    return True

tasks = [
    "clean_detection",
    "compressed_detection",
    "adversarial_detection",
    "streaming_detection",
    "phonecall_detection",
]

violations = []

# ── Test 1: Full 5-step protocol (normal agent) ────────────────────────
print("=== Test 1: Full 5-step protocol ===")
for task in tasks:
    print(f"\n  Task: {task}")
    resp = reset(task)
    r = resp.get("reward", 0)
    if not check_reward(r, f"reset {task}"):
        violations.append(f"reset {task}: {r}")

    rewards = []
    for i, act in enumerate([
        {"action_type": "request_temporal_features"},
        {"action_type": "request_spectral_features"},
        {"action_type": "request_comparison"},
        {"action_type": "analyze_evidence"},
        {"action_type": "final_classify", "label": 0, "confidence": 0.7,
         "reasoning": "human speech with natural jitter and shimmer variation"},
    ]):
        resp = step(act, task)
        r = resp["reward"]
        rewards.append(r)
        if not check_reward(r, f"step {i+1} ({act['action_type']}) task={task}"):
            violations.append(f"step {i+1} {task}: {r}")
    print(f"  rewards: {rewards}")

# ── Test 2: Jump straight to classify (worst case) ─────────────────────
print("\n=== Test 2: Jump to final_classify (no exploration) ===")
for task in tasks:
    print(f"\n  Task: {task}")
    reset(task, seed=42)
    
    # Try both labels
    for label in [0, 1]:
        reset(task, seed=42)
        resp = step({
            "action_type": "final_classify",
            "label": label,
            "confidence": 0.99,
            "reasoning": ""
        }, task)
        r = resp["reward"]
        if not check_reward(r, f"jump-classify label={label} task={task}"):
            violations.append(f"jump {task} label={label}: {r}")
        print(f"  label={label} reward={r}")

# ── Test 3: Edge confidence values ─────────────────────────────────────
print("\n=== Test 3: Edge confidence values ===")
for task in tasks:
    for conf in [0.0, 0.001, 0.5, 0.999, 1.0]:
        reset(task, seed=7)
        resp = step({
            "action_type": "final_classify",
            "label": 0,
            "confidence": conf,
            "reasoning": "test"
        }, task)
        r = resp["reward"]
        if not check_reward(r, f"conf={conf} task={task}"):
            violations.append(f"conf {task} conf={conf}: {r}")
        print(f"  {task} conf={conf}: reward={r}")

# ── Test 4: Various seeds to trigger different samples ─────────────────
print("\n=== Test 4: Multiple seeds (checking sample variation) ===")
for task in tasks:
    for seed in [0, 1, 2, 3, 42, 100, 999]:
        reset(task, seed=seed)
        # Minimal exploration + classify
        step({"action_type": "request_temporal_features"}, task)
        resp = step({
            "action_type": "final_classify",
            "label": 1,
            "confidence": 0.6,
            "reasoning": "synthetic fake generated smooth"
        }, task)
        r = resp["reward"]
        if not check_reward(r, f"seed={seed} task={task}"):
            violations.append(f"seed {task} seed={seed}: {r}")

print(f"\n\n{'='*60}")
if violations:
    print(f"FOUND {len(violations)} VIOLATIONS:")
    for v in violations:
        print(f"  - {v}")
else:
    print("ALL SCORES STRICTLY IN (0, 1) - NO VIOLATIONS FOUND")
print(f"{'='*60}")
