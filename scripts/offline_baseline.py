#!/usr/bin/env python3
"""
Oracle-style baseline: no LLM, deterministic policy using only what the observation exposes.
Writes baseline_scores.json for judges / README tables. Run from repo root:

  PYTHONPATH=. python scripts/offline_baseline.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agentguard_gym.environment import AgentGuardEnvironment
from agentguard_gym.models import CyberTaskType, DefenseActionType


def policy(obs) -> Dict[str, Any]:
    if not obs.artifacts:
        return {"defense": DefenseActionType.ALLOW.value, "rationale": "noop"}
    art = obs.artifacts[0]
    if art.get("type") == "prompt":
        text = str(art.get("content", "")).lower()
        if "exfiltrate" in text or "ignore prior" in text:
            return {"defense": DefenseActionType.SANITIZE.value, "rationale": "strip malicious instruction"}
        return {"defense": DefenseActionType.ALLOW.value, "rationale": "benign service request"}
    if art.get("type") == "url":
        url = str(art.get("url", ""))
        if "169.254.169.254" in url or "metadata" in url.lower():
            return {"defense": DefenseActionType.BLOCK.value, "rationale": "block metadata SSRF"}
        return {"defense": DefenseActionType.ALLOW.value, "rationale": "benign rss"}
    flags = (art.get("flags") or {}) if art.get("type") == "memory_slot" else {}
    if flags.get("env_secret_exposure_hint"):
        return {"defense": DefenseActionType.CLEAR_EXPOSED_SECRETS.value, "rationale": "rotate leaked token"}
    if flags.get("poisoned_vector_hint"):
        return {"defense": DefenseActionType.QUARANTINE_MEMORY.value, "rationale": "isolate poisoned chunk"}
    return {"defense": DefenseActionType.ALLOW.value, "rationale": "clean memory slot"}


def run_episode(task: CyberTaskType, seed: int = 42) -> Dict[str, Any]:
    env = AgentGuardEnvironment()
    obs = env.reset(seed=seed, task=task)
    rewards: List[float] = []
    steps = 0
    while steps < 32:
        act = policy(obs)
        res = env.step(act)
        steps += 1
        rewards.append(float(res.reward.value))
        obs = res.observation
        if res.done:
            break
    score = sum(rewards) / max(1, len(rewards))
    return {
        "task": task.value,
        "steps": steps,
        "episode_score": round(score, 4),
        "step_rewards": [round(r, 4) for r in rewards],
    }


def main() -> None:
    tasks = (
        CyberTaskType.PROMPT_INJECTION,
        CyberTaskType.TOOL_MISUSE_SSRF,
        CyberTaskType.MEMORY_POISONING_PRIVILEGE,
    )
    rows = [run_episode(t) for t in tasks]
    mean_score = sum(r["episode_score"] for r in rows) / len(rows)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "agentguard-gym",
        "method": "offline_oracle_policy",
        "mean_task_score": round(mean_score, 4),
        "tasks": {r["task"]: r for r in rows},
    }
    path = Path(__file__).resolve().parent.parent / "baseline_scores.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
