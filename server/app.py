"""
server.py — FastAPI HTTP server for the ASP Coordinator Environment.

Exposes the OpenEnv standard REST endpoints:
  POST /reset          → start new episode, returns Observation
  POST /step           → take one action, returns (obs, reward, done, info)
  GET  /state          → current episode state snapshot
  GET  /tasks          → list all tasks with difficulty labels
  GET  /health         → health check (ping endpoint for HF Spaces)
  POST /grade          → run the deterministic grader, returns score

The inference script (inference.py) calls these endpoints via the OpenAI client.
Judges ping /health to verify the Space is alive.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.asp_env import ASPEnv
from env.models import Action, ActionType
from tasks.tasks import ALL_TASKS, list_tasks

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hospital ASP Coordinator Environment",
    description=(
        "An OpenEnv-compliant reinforcement learning environment simulating "
        "a hospital Antimicrobial Stewardship Program pharmacist/physician. "
        "The agent manages antibiotic prescription reviews with four competing "
        "objectives: patient safety, resistance stewardship, cost efficiency, "
        "and prescriber satisfaction."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task (simple; production would use session IDs)
_envs: Dict[str, ASPEnv] = {}


def _get_env(task_id: str) -> ASPEnv:
    if task_id not in _envs:
        _envs[task_id] = ASPEnv(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_1"


class StepRequest(BaseModel):
    task_id: str = "task_1"
    action_type: str
    parameters: Dict[str, Any] = {}


class GradeRequest(BaseModel):
    task_id: str = "task_1"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check — judges ping this to verify the Space is alive."""
    return {"status": "ok", "environment": "Hospital ASP Coordinator"}


@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty,
                "description_preview": t.description[:120] + "...",
            }
            for t in ALL_TASKS.values()
        ]
    }


@app.post("/reset")
def reset(body: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """Start a new episode for the given task. Returns initial Observation."""
    if body is None:
        body = ResetRequest()
    if body.task_id not in ALL_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{body.task_id}'. Available: {list_tasks()}",
        )
    env = _get_env(body.task_id)
    obs, info = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(body: StepRequest) -> Dict[str, Any]:
    """Execute one action. Returns (observation, reward, done, info)."""
    if body.task_id not in ALL_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{body.task_id}'")

    # Validate action_type
    try:
        atype = ActionType(body.action_type)
    except ValueError:
        valid = [e.value for e in ActionType]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{body.action_type}'. Valid: {valid}",
        )

    env = _get_env(body.task_id)
    action = Action(action_type=atype, parameters=body.parameters)

    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": terminated or truncated,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = "task_1") -> Dict[str, Any]:
    """Return current full state snapshot (for debugging and openenv validate)."""
    if task_id not in ALL_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_id}'")
    env = _get_env(task_id)
    return env.state()


@app.post("/grade")
def grade(body: Optional[GradeRequest] = None) -> Dict[str, Any]:
    """
    Run the deterministic grader for the current episode state.
    Returns score 0.0–1.0.
    """
    if body is None:
        body = GradeRequest()
    if body.task_id not in ALL_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{body.task_id}'")
    env = _get_env(body.task_id)
    score = env.run_grader()
    return {
        "task_id": body.task_id,
        "score": score,
        "difficulty": ALL_TASKS[body.task_id].difficulty,
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
