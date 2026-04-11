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


@app.get("/web", response_class=HTMLResponse)
def web_interface():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Authenticity OpenEnv</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, sans-serif; max-width: 860px; margin: 50px auto; padding: 20px; background: #050508; color: #fff; }
        h1 { color: #00c9a7; font-size: 28px; margin-bottom: 8px; }
        h2 { font-size: 16px; font-weight: 500; margin-bottom: 12px; color: #00c9a7; }
        p { color: #666; font-size: 14px; line-height: 1.6; margin-bottom: 8px; }
        .card { background: #080810; border: 1px solid #0f0f1a; border-radius: 14px; padding: 20px; margin: 16px 0; }
        .tag { background: #0d2d1e; color: #00c9a7; padding: 4px 12px; border-radius: 20px; font-size: 11px; margin: 3px; display: inline-block; border: 1px solid #0f2d26; }
        a { color: #00c9a7; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .task { border-left: 2px solid #00c9a7; padding: 8px 12px; margin: 8px 0; background: #050508; border-radius: 0 8px 8px 0; }
        .task strong { font-size: 13px; color: #fff; }
        .task span { font-size: 12px; color: #555; display: block; margin-top: 2px; }
        .difficulty { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; margin-left: 8px; }
        .easy { background: #0d2d1e; color: #00c9a7; }
        .medium { background: #1a1a00; color: #f0a500; }
        .hard { background: #1a0000; color: #ff6b6b; }
        .extreme { background: #1a0010; color: #ff00aa; }
        .medium_hard { background: #0d1a2d; color: #00aaff; }
        .endpoint { display: flex; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid #0f0f1a; }
        .endpoint:last-child { border-bottom: none; }
        .method { font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px; min-width: 45px; text-align: center; }
        .get { background: #0d2d1e; color: #00c9a7; }
        .post { background: #1a1a00; color: #f0a500; }
        .endpoint-path { font-size: 13px; color: #fff; font-family: monospace; }
        .endpoint-desc { font-size: 12px; color: #444; }
        .action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
        .action-card { background: #050508; border: 1px solid #0f0f1a; border-radius: 10px; padding: 12px; }
        .action-name { font-size: 12px; font-family: monospace; color: #00c9a7; margin-bottom: 4px; }
        .action-desc { font-size: 11px; color: #444; line-height: 1.5; }
        .stat { text-align: center; padding: 16px; }
        .stat-num { font-size: 28px; font-weight: 600; color: #fff; }
        .stat-num span { color: #00c9a7; }
        .stat-label { font-size: 11px; color: #444; margin-top: 4px; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #0f0f1a; border-radius: 12px; overflow: hidden; }
        .stat { background: #080810; }
        .badge { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px; border: 1px solid #0f2d26; background: #050f0d; border-radius: 20px; font-size: 11px; color: #00c9a7; }
        .dot { width: 6px; height: 6px; background: #00c9a7; border-radius: 50%; animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
        footer { text-align: center; padding: 2rem 0; color: #333; font-size: 12px; }
        footer span { color: #00c9a7; }
    </style>
</head>
<body>
    <div style="margin-bottom:1.5rem">
        <div class="badge"><div class="dot"></div>Live — 5 tasks available</div>
    </div>

    <h1>🎙️ Voice Authenticity OpenEnv</h1>
    <p style="margin-bottom:1.5rem;font-size:16px;color:#888">
        Multi-step agentic environment for detecting synthetic (AI-generated) speech
        across real-world degradation and adversarial conditions.
    </p>

    <div class="stats-grid">
        <div class="stat">
            <div class="stat-num">5<span>+</span></div>
            <div class="stat-label">Tasks</div>
        </div>
        <div class="stat">
            <div class="stat-num">5</div>
            <div class="stat-label">Steps per episode</div>
        </div>
        <div class="stat">
            <div class="stat-num">48</div>
            <div class="stat-label">Feature dimensions</div>
        </div>
    </div>

    <div class="card">
        <h2>Tasks</h2>
        <div class="task">
            <strong>clean_detection <span class="difficulty easy">easy</span></strong>
            <span>Classify real vs synthetic speech from clean, unmodified audio features</span>
        </div>
        <div class="task">
            <strong>compressed_detection <span class="difficulty medium">medium</span></strong>
            <span>Classify speech under codec compression degradation</span>
        </div>
        <div class="task">
            <strong>adversarial_detection <span class="difficulty hard">hard</span></strong>
            <span>Adversarially crafted synthetic speech with overlapping feature distributions</span>
        </div>
        <div class="task">
            <strong>streaming_detection <span class="difficulty medium_hard">medium-hard</span></strong>
            <span>Step-dependent noise soft-gating — earlier steps noisier, later steps cleaner</span>
        </div>
        <div class="task">
            <strong>phonecall_detection <span class="difficulty extreme">extreme</span></strong>
            <span>Heavy codec compression and narrowband degradation simulating phone calls</span>
        </div>
    </div>

    <div class="card">
        <h2>5-Step Agent Protocol</h2>
        <div class="action-grid">
            <div class="action-card">
                <div class="action-name">1. request_temporal_features</div>
                <div class="action-desc">Reveals jitter, shimmer, and HNR — the core discriminating signals</div>
            </div>
            <div class="action-card">
                <div class="action-name">2. request_spectral_features</div>
                <div class="action-desc">Reveals 20 MFCC means, 20 MFCC stds, ZCR, spectral centroid</div>
            </div>
            <div class="action-card">
                <div class="action-name">3. request_comparison</div>
                <div class="action-desc">Compares sample to real/fake reference centroids via cosine similarity</div>
            </div>
            <div class="action-card">
                <div class="action-name">4. analyze_evidence</div>
                <div class="action-desc">Synthesizes all gathered signals into a structured evidence summary</div>
            </div>
            <div class="action-card" style="grid-column: span 2;">
                <div class="action-name">5. final_classify</div>
                <div class="action-desc">Submits final verdict: label (0=real, 1=synthetic) + confidence + reasoning. Terminates episode.</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>API Endpoints</h2>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="endpoint-path">/reset</span>
            <span class="endpoint-desc">Reset episode, optionally set task_name</span>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="endpoint-path">/step</span>
            <span class="endpoint-desc">Submit action, receive observation + reward</span>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="endpoint-path">/state</span>
            <span class="endpoint-desc">Current environment state</span>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="endpoint-path">/health</span>
            <span class="endpoint-desc">Health check</span>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="endpoint-path"><a href="/docs">/docs</a></span>
            <span class="endpoint-desc">Interactive API documentation (Swagger UI)</span>
        </div>
    </div>

    <div class="card">
        <h2>Tags</h2>
        <span class="tag">openenv</span>
        <span class="tag">speech</span>
        <span class="tag">fraud-detection</span>
        <span class="tag">audio</span>
        <span class="tag">partial-observability</span>
        <span class="tag">multi-step</span>
        <span class="tag">confidence-calibration</span>
        <span class="tag">adversarial</span>
    </div>

    <footer>
        Built by <span>Akshara Sharma</span> · Voice Authenticity OpenEnv v2.0.0
        · <a href="https://github.com/AksharaaSharmaa/voice-authenticity-openenv">GitHub</a>
    </footer>
</body>
</html>
"""


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


@app.get("/")
def root():
    return {
        "name": "voice-authenticity-openenv",
        "version": "2.0.0",
        "status": "running",
        "tasks": TASKS,
        "web": "/web",
        "docs": "/docs"
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()