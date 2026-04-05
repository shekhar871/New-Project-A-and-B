"""Thin synchronous HTTP wrapper — handy when the FastAPI server lives on another port."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from agentguard_gym.models import AgentGuardObservation, AgentGuardReward, CyberTaskType, StepResult


class AgentGuardGymClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AgentGuardGymClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[CyberTaskType] = None,
    ) -> AgentGuardObservation:
        r = self._client.post(
            f"{self._base}/reset",
            json={"seed": seed, "episode_id": episode_id, "task": task.value if task else None},
        )
        r.raise_for_status()
        data = r.json()
        return AgentGuardObservation.model_validate(data["observation"])

    def step(self, action: Dict[str, Any]) -> StepResult:
        r = self._client.post(f"{self._base}/step", json={"action": action})
        r.raise_for_status()
        data = r.json()
        reward_payload = data["reward"]
        if isinstance(reward_payload, dict):
            reward = AgentGuardReward.model_validate(reward_payload)
        else:
            reward = AgentGuardReward(value=float(reward_payload), utility_raw=float(reward_payload))
        return StepResult(
            observation=AgentGuardObservation.model_validate(data["observation"]),
            reward=reward,
            done=bool(data["done"]),
            info=data.get("info") or {},
        )

    def state(self) -> Dict[str, Any]:
        r = self._client.get(f"{self._base}/state")
        r.raise_for_status()
        return r.json()
