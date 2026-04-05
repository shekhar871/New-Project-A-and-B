from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from aml_defense_gym.models import AMLObservation, AMLReward, AMLTaskType, StepResult


class AMLDefenseGymClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AMLDefenseGymClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[AMLTaskType] = None,
    ) -> AMLObservation:
        payload: Dict[str, Any] = {"seed": seed, "episode_id": episode_id}
        if task is not None:
            payload["task"] = task.value
        r = self._client.post(f"{self._base}/reset", json=payload)
        r.raise_for_status()
        data = r.json()
        return AMLObservation.model_validate(data["observation"])

    def step(self, action: Dict[str, Any]) -> StepResult:
        r = self._client.post(f"{self._base}/step", json={"action": action})
        r.raise_for_status()
        data = r.json()
        reward = AMLReward.model_validate(data["reward"])
        return StepResult(
            observation=AMLObservation.model_validate(data["observation"]),
            reward=reward,
            done=bool(data["done"]),
            info=data.get("info") or {},
        )

    def state(self) -> Dict[str, Any]:
        r = self._client.get(f"{self._base}/state")
        r.raise_for_status()
        return r.json()
