"""
FastAPI entrypoint expected by the OpenEnv hackathon layout (`server/app.py` next to the package).

We keep one process-wide environment for the simplest HTTP smoke tests; swap in a session
factory if you need concurrent WebSocket classrooms later.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agentguard_gym.environment import AgentGuardEnvironment
from agentguard_gym.models import CyberTaskType

app = FastAPI(title="AgentGuard-Gym", version="0.2.0")
_env = AgentGuardEnvironment()


class ResetBody(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = None
    task: Optional[CyberTaskType] = None


class StepBody(BaseModel):
    action: Dict[str, Any]


@app.post("/reset")
def http_reset(body: ResetBody) -> Dict[str, Any]:
    obs = _env.reset(seed=body.seed, episode_id=body.episode_id, task=body.task)
    return {"observation": obs.model_dump(mode="json"), "reward": None, "done": False}


@app.post("/step")
def http_step(body: StepBody) -> Dict[str, Any]:
    result = _env.step(body.action)
    return {
        "observation": result.observation.model_dump(mode="json"),
        "reward": result.reward.model_dump(mode="json"),
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def http_state() -> Dict[str, Any]:
    return _env.state().model_dump(mode="json")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


def main() -> None:
    """CLI entry used by `uv run server` / OpenEnv validators (binds PORT from env, default 8000)."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
