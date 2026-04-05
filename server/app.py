from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import VoiceAuthenticityEnv

app = FastAPI(title="Voice Authenticity OpenEnv")

envs = {
    "clean_detection":       VoiceAuthenticityEnv("clean_detection"),
    "compressed_detection":  VoiceAuthenticityEnv("compressed_detection"),
    "adversarial_detection": VoiceAuthenticityEnv("adversarial_detection"),
}

current_task = "clean_detection"

class ActionRequest(BaseModel):
    label: Optional[int] = 0
    confidence: Optional[float] = 0.5
    reasoning: Optional[str] = ""
    task_name: Optional[str] = None

@app.post("/reset")
def reset(request: dict = {}):
    global current_task
    task = request.get("task_name", current_task) if request else current_task
    if task not in envs:
        task = "clean_detection"
    current_task = task
    obs = envs[current_task].reset()
    return JSONResponse({
        "observation": obs.dict(),
        "done": False,
        "reward": 0.0,
        "info": {}
    })

@app.post("/step")
def step(action: ActionRequest):
    global current_task
    task = action.task_name or current_task
    if task not in envs:
        task = current_task
    action_dict = {
        "label": action.label,
        "confidence": action.confidence,
        "reasoning": action.reasoning
    }
    obs, reward, done, info = envs[task].step(action_dict)
    return JSONResponse({
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    })

@app.get("/state")
def state():
    return JSONResponse(envs[current_task].state())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "voice-authenticity-openenv", "status": "running"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()