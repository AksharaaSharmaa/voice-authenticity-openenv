from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import textwrap
import json
import requests
from typing import List, Optional
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "voice-authenticity"
MAX_STEPS = 1
SUCCESS_SCORE_THRESHOLD = 0.5

# Environment server URL
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert audio forensics agent detecting synthetic (AI-generated) speech.
You receive a 48-dimensional normalized feature vector AND key raw values in the hint.

Always use the KEY VALUES in the hint for classification:

REAL speech thresholds (from dataset):
- jitter > 0.025
- shimmer > 0.10
- hnr < 12.0

SYNTHETIC speech thresholds:
- jitter < 0.020
- shimmer < 0.09
- hnr > 12.0

When in doubt, lower your confidence. Never exceed 0.85 confidence on hard tasks.

Respond ONLY with valid JSON:
{"label": 0 or 1, "confidence": 0.0-1.0, "reasoning": "brief"}
0 = real human speech
1 = synthetic/AI-generated speech
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def env_reset(task_name: str) -> dict:
    """Call /reset on the environment server."""
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={"task_name": task_name},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def env_step(action: dict, task_name: str) -> dict:
    """Call /step on the environment server."""
    payload = {
        "label": action["label"],
        "confidence": action["confidence"],
        "reasoning": action.get("reasoning", ""),
        "task_name": task_name
    }
    response = requests.post(
        f"{ENV_SERVER_URL}/step",
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def get_agent_action(client, observation: dict) -> dict:
    features = observation.get("features", [])
    task_name = observation.get("task_name", "")
    difficulty = observation.get("difficulty", "")
    hint = observation.get("hint", "")

    user_prompt = f"""
Audio sample features: {features}
Task: {task_name} (difficulty: {difficulty})
{f'Note: {hint}' if hint else ''}

Classify this audio sample. Keep reasoning under 70 words. Respond with JSON only.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0.3,
            max_tokens=250,
            stream=False
        )
        text = completion.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        last_brace = text.rfind("}")
        if last_brace != -1:
            text = text[:last_brace + 1]
        result = json.loads(text)
        result["label"] = int(result.get("label", 0))
        result["confidence"] = float(result.get("confidence", 0.5))
        result["label"] = result["label"] if result["label"] in [0, 1] else 0
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        return result
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {"label": 0, "confidence": 0.5, "reasoning": "fallback"}


async def run_task(client: OpenAI, task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment via HTTP
        reset_response = env_reset(task_name)
        observation = reset_response.get("observation", {})
        done = reset_response.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM
            action_dict = get_agent_action(client, observation)
            action_str = json.dumps(action_dict)

            # Step environment via HTTP
            step_response = env_step(action_dict, task_name)
            observation = step_response.get("observation", {})
            reward = float(step_response.get("reward", 0.0))
            done = step_response.get("done", True)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                    done=done, error=error)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        score_val = sum(rewards) / len(rewards) if rewards else 0.0
        log_end(success=success, steps=steps_taken,
               score=score_val, rewards=rewards)


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["clean_detection", "compressed_detection", "adversarial_detection"]
    for task in tasks:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())