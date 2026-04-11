"""Exhaustive local test of ALL grader paths to find 0.0 or 1.0 scores."""
from environment.graders import grade
from environment.env import VoiceAuthenticityEnv, TASKS, DIFFICULTY_MAP

violations = []
total = 0

difficulties = ["easy", "medium", "medium_hard", "hard", "extreme"]

# All possible action histories
action_histories = [
    ["final_classify"],
    ["request_temporal_features", "final_classify"],
    ["request_spectral_features", "final_classify"],
    ["request_comparison", "final_classify"],
    ["analyze_evidence", "final_classify"],
    ["request_temporal_features", "request_spectral_features", "final_classify"],
    ["request_temporal_features", "request_spectral_features", "request_comparison", "final_classify"],
    ["request_temporal_features", "request_spectral_features", "request_comparison", "analyze_evidence", "final_classify"],
    ["request_temporal_features", "analyze_evidence", "final_classify"],
    ["analyze_evidence", "request_temporal_features", "final_classify"],
]

labels = [0, 1]
true_labels = [0, 1]
confidences = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
reasonings = [
    "",
    "test",
    "real human natural jitter",
    "synthetic fake generated smooth",
    "real but also synthetic",
    "no keywords here at all just random text padding to exceed minimum length",
]

for diff in difficulties:
    for tl in true_labels:
        for pl in labels:
            for conf in confidences:
                for reasoning in reasonings:
                    for history in action_histories:
                        action = {"label": pl, "confidence": conf, "reasoning": reasoning}
                        result = grade(tl, action, diff, history)
                        score = result["score"]
                        total += 1
                        if score <= 0.0 or score >= 1.0:
                            violations.append({
                                "score": score,
                                "true_label": tl,
                                "pred_label": pl,
                                "confidence": conf,
                                "difficulty": diff,
                                "reasoning": reasoning[:30],
                                "history": history,
                            })

# Also test via the environment step() directly
print("Testing via environment step()...")
env_violations = []
for task in TASKS:
    env = VoiceAuthenticityEnv(task)
    for seed in range(20):
        env.reset(seed=seed)
        
        # Test jump-to-classify
        for label in [0, 1]:
            for conf in [0.0, 0.5, 1.0]:
                env.reset(seed=seed)
                obs, reward, done, info = env.step({
                    "action_type": "final_classify",
                    "label": label,
                    "confidence": conf,
                    "reasoning": "test reasoning text"
                })
                total += 1
                if reward <= 0.0 or reward >= 1.0:
                    env_violations.append(f"task={task} seed={seed} label={label} conf={conf} reward={reward}")

        # Test full protocol
        env.reset(seed=seed)
        obs, r1, _, _ = env.step({"action_type": "request_temporal_features"})
        total += 1
        if r1 <= 0.0 or r1 >= 1.0:
            env_violations.append(f"temporal task={task} seed={seed} reward={r1}")
        
        obs, r2, _, _ = env.step({"action_type": "request_spectral_features"})
        total += 1
        if r2 <= 0.0 or r2 >= 1.0:
            env_violations.append(f"spectral task={task} seed={seed} reward={r2}")
        
        obs, r3, _, _ = env.step({"action_type": "request_comparison"})
        total += 1
        if r3 <= 0.0 or r3 >= 1.0:
            env_violations.append(f"comparison task={task} seed={seed} reward={r3}")
        
        obs, r4, _, _ = env.step({"action_type": "analyze_evidence"})
        total += 1
        if r4 <= 0.0 or r4 >= 1.0:
            env_violations.append(f"analyze task={task} seed={seed} reward={r4}")
        
        obs, r5, done, info = env.step({
            "action_type": "final_classify",
            "label": 0, "confidence": 0.7,
            "reasoning": "natural speech with jitter variation"
        })
        total += 1
        if r5 <= 0.0 or r5 >= 1.0:
            env_violations.append(f"classify task={task} seed={seed} reward={r5}")

print(f"\nTested {total} combinations")
print(f"\nGrader violations: {len(violations)}")
for v in violations[:20]:
    print(f"  {v}")
print(f"\nEnv step violations: {len(env_violations)}")
for v in env_violations[:20]:
    print(f"  {v}")

if not violations and not env_violations:
    print("\nALL SCORES STRICTLY IN (0, 1) - PASS")
else:
    print("\nFAILED - found violations!")
