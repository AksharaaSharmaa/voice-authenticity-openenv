from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os

from environment.env import VoiceAuthenticityEnv

app = FastAPI(
    title="Voice Authenticity OpenEnv",
    description="Multi-step agentic environment for detecting synthetic speech",
    version="2.0.0"
)

TASKS = [
    "clean_detection",
    "compressed_detection",
    "adversarial_detection",
    "streaming_detection",
    "phonecall_detection",
]

envs = {task: VoiceAuthenticityEnv(task) for task in TASKS}
current_task = "clean_detection"


class ActionRequest(BaseModel):
    action_type: str = "final_classify"
    label: int = 0
    confidence: float = 0.5
    reasoning: str = ""
    focus: List[str] = []
    task_name: Optional[str] = None


# ── Serve Dashboard.html at /web ────────────────────────────────────────

_dashboard_html = None

def _load_dashboard():
    global _dashboard_html
    if _dashboard_html is None:
        html_path = os.path.join(os.path.dirname(__file__), "Dashboard.html")
        with open(html_path, "r", encoding="utf-8") as f:
            _dashboard_html = f.read()
    return _dashboard_html


@app.get("/", response_class=HTMLResponse)
def root():
    return _load_dashboard()


@app.get("/web", response_class=HTMLResponse)
def web_interface():
    return _load_dashboard()


# ── API Endpoints ───────────────────────────────────────────────────────

@app.post("/reset")
def reset(request: dict = {}):
    global current_task
    task = request.get("task_name", current_task) if request else current_task
    if task not in envs:
        task = "clean_detection"
    current_task = task
    seed = request.get("seed") if request else None
    obs = envs[current_task].reset(seed=seed)
    return JSONResponse({
        "observation": obs.dict(),
        "done": False,
        "reward": 0.05,
        "info": {}
    })


@app.post("/step")
def step(action: ActionRequest):
    global current_task
    task = action.task_name or current_task
    if task not in envs:
        task = current_task
    action_dict = {
        "action_type": action.action_type,
        "label": action.label,
        "confidence": action.confidence,
        "reasoning": action.reasoning,
        "focus": action.focus,
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
    return {"status": "healthy", "service": "voice-authenticity-openenv"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()