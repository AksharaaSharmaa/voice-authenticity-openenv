"""
Baseline inference agent for Voice Authenticity OpenEnv v2.

Uses the 5-action protocol:
  1. request_temporal_features  → get jitter, shimmer, HNR
  2. request_spectral_features  → get MFCC values
  3. request_comparison         → get similarity to real/fake centroids
  4. analyze_evidence           → synthesize gathered information
  5. final_classify             → submit label + confidence + reasoning
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import textwrap
import json
import requests
from typing import List
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "voice-authenticity"
SUCCESS_SCORE_THRESHOLD = 0.60

# Environment server URL
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert audio forensics agent detecting synthetic (AI-generated) speech.
You operate in a multi-step environment where you must gather evidence before classifying.

REAL speech indicators:
- jitter > 0.025 (vocal cord irregularity)
- shimmer > 0.10 (amplitude variation)
- HNR < 12.0 (more noise in signal)
- Higher MFCC std deviations (natural variation)

SYNTHETIC speech indicators:
- jitter < 0.020 (too stable)
- shimmer < 0.09 (too uniform)
- HNR > 12.0 (too clean)
- Lower MFCC std deviations (artificial consistency)

COMPARISON interpretation:
- Higher cosine similarity to real centroid → likely real
- Higher cosine similarity to fake centroid → likely synthetic
- Closer euclidean distance to real → likely real

CONFIDENCE GUIDELINES:
- Easy tasks: confident predictions okay (0.7-0.9)
- Medium tasks: moderate confidence (0.6-0.8)
- Hard/extreme tasks: calibrate carefully, never exceed 0.85

Respond ONLY with valid JSON for the requested action type.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    if isinstance(action, dict):
        action_type = action.get("action_type", "")
        if action_type == "final_classify":
            label = action.get("label", 0)
            conf = action.get("confidence", 0.5)
            action_str = f"final_classify label={label} confidence={conf:.2f}"
        else:
            action_str = action_type or str(action)
    else:
        action_str = str(action)
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards, grader_breakdown=None):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    breakdown_str = ""
    if grader_breakdown:
        breakdown_str = (
            f' grader_breakdown={{'
            f'"correctness":{grader_breakdown.get("correctness", 0):.2f},'
            f'"calibration":{grader_breakdown.get("confidence_calibration", 0):.2f},'
            f'"trajectory":{grader_breakdown.get("trajectory_quality", 0):.2f},'
            f'"utilization":{grader_breakdown.get("feature_utilization", 0):.2f},'
            f'"reasoning":{grader_breakdown.get("reasoning_consistency", 0):.2f},'
            f'"ordering":{grader_breakdown.get("action_ordering", 0):.2f}}}'
        )
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}{breakdown_str}",
        flush=True,
    )


# ── Environment API calls ──────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    """Call /reset on the environment server."""
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={"task_name": task_name, "seed": 7},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(action: dict, task_name: str) -> dict:
    """Call /step on the environment server."""
    payload = {
        "action_type": action.get("action_type", "final_classify"),
        "label": action.get("label", 0),
        "confidence": action.get("confidence", 0.5),
        "reasoning": action.get("reasoning", ""),
        "task_name": task_name,
    }
    response = requests.post(
        f"{ENV_SERVER_URL}/step",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ── LLM agent decision making ──────────────────────────────────────────

def get_classification(client, context: dict) -> dict:
    """Ask the LLM to make a final classification based on accumulated evidence."""
    user_prompt = f"""
Based on the following evidence gathered from an audio sample, classify it as
real (0) or synthetic (1). Task: {context['task_name']} (difficulty: {context['difficulty']})

EVIDENCE GATHERED:
{json.dumps(context.get('visible_features', {}), indent=2)}

COMPARISON RESULTS:
{json.dumps(context.get('comparison_result', {}), indent=2)}

ANALYSIS SUMMARY:
{context.get('evidence_summary', 'No analysis performed.')}

ACTIONS TAKEN: {', '.join(context.get('actions_taken', []))}

Respond with JSON only:
{{"label": 0 or 1, "confidence": 0.0-1.0, "reasoning": "brief explanation under 70 words"}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.3,
            max_tokens=300,
            stream=False,
        )
        text = completion.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        last_brace = text.rfind("}")
        if last_brace != -1:
            text = text[: last_brace + 1]
        result = json.loads(text)
        result["label"] = int(result.get("label", 0))
        result["confidence"] = float(result.get("confidence", 0.5))
        result["label"] = result["label"] if result["label"] in [0, 1] else 0
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        return result
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {"label": 0, "confidence": 0.5, "reasoning": "fallback"}


# ── Main task runner ────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_name: str):
    """Run one episode of a task using the 5-action protocol."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    context = {}
    grader_breakdown = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ───────────────────────────────────────────
        reset_response = env_reset(task_name)
        observation = reset_response.get("observation", {})
        context = {
            "task_name": observation.get("task_name", task_name),
            "difficulty": observation.get("difficulty", ""),
            "visible_features": {},
            "comparison_result": None,
            "evidence_summary": None,
            "actions_taken": [],
        }

        # ── Step 1: Request temporal features ───────────────
        action1 = {"action_type": "request_temporal_features"}
        step1 = env_step(action1, task_name)
        observation = step1.get("observation", {})
        reward1 = float(step1.get("reward", 0.0))
        rewards.append(reward1)
        steps_taken = 1
        context["visible_features"] = observation.get("visible_features", {})
        context["actions_taken"] = observation.get("actions_taken", [])

        log_step(step=1, action=action1, reward=reward1,
                 done=step1.get("done", False), error=None)

        if step1.get("done", False):
            raise RuntimeError("Episode ended prematurely at step 1")

        # ── Step 2: Request spectral features ───────────────
        action2 = {"action_type": "request_spectral_features"}
        step2 = env_step(action2, task_name)
        observation = step2.get("observation", {})
        reward2 = float(step2.get("reward", 0.0))
        rewards.append(reward2)
        steps_taken = 2
        context["visible_features"] = observation.get("visible_features", {})
        context["actions_taken"] = observation.get("actions_taken", [])

        log_step(step=2, action=action2, reward=reward2,
                 done=step2.get("done", False), error=None)

        if step2.get("done", False):
            raise RuntimeError("Episode ended prematurely at step 2")

        # ── Step 3: Request comparison ──────────────────────
        action3 = {"action_type": "request_comparison"}
        step3 = env_step(action3, task_name)
        observation = step3.get("observation", {})
        reward3 = float(step3.get("reward", 0.0))
        rewards.append(reward3)
        steps_taken = 3
        context["visible_features"] = observation.get("visible_features", {})
        context["comparison_result"] = observation.get("comparison_result", {})
        context["actions_taken"] = observation.get("actions_taken", [])

        log_step(step=3, action=action3, reward=reward3,
                 done=step3.get("done", False), error=None)

        if step3.get("done", False):
            raise RuntimeError("Episode ended prematurely at step 3")

        # ── Step 4: Analyze evidence ────────────────────────
        action4 = {"action_type": "analyze_evidence"}
        step4 = env_step(action4, task_name)
        observation = step4.get("observation", {})
        reward4 = float(step4.get("reward", 0.0))
        rewards.append(reward4)
        steps_taken = 4
        context["evidence_summary"] = observation.get("evidence_summary", "")
        context["actions_taken"] = observation.get("actions_taken", [])

        log_step(step=4, action=action4, reward=reward4,
                 done=step4.get("done", False), error=None)

        if step4.get("done", False):
            raise RuntimeError("Episode ended prematurely at step 4")

        # ── Step 5: Final classify (LLM decision) ──────────
        classification = get_classification(client, context)
        action5 = {
            "action_type": "final_classify",
            "label": classification["label"],
            "confidence": classification["confidence"],
            "reasoning": classification.get("reasoning", ""),
        }

        step5 = env_step(action5, task_name)
        reward5 = float(step5.get("reward", 0.0))
        rewards.append(reward5)
        steps_taken = 5
        done = step5.get("done", True)

        # Capture grader breakdown from environment info
        step5_info = step5.get("info", {})
        grader_breakdown = step5_info.get("grader_breakdown", None)

        log_step(step=5, action=action5, reward=reward5,
                 done=done, error=None)

        # Score is the final classify reward (main grader score)
        score = reward5
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        # The competition judges score based on the final classify reward only
        if rewards:
            score = rewards[-1]
        else:
            score = 0.0

        log_end(
            success=success, steps=steps_taken,
            score=score, rewards=rewards,
            grader_breakdown=grader_breakdown,
        )
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [
        "clean_detection",
        "compressed_detection",
        "adversarial_detection",
        "streaming_detection",
        "phonecall_detection",
    ]
    for task in tasks:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())