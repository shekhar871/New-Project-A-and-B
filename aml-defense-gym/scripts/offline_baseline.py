#!/usr/bin/env python3
"""
Deterministic AML oracle (no LLM): thresholds on screening scores, full EDD stub, perfect TM ranking
when labels appear in records (synthetic CSV includes is_laundering).

  PYTHONPATH=. python scripts/offline_baseline.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aml_defense_gym.environment import AMLDefenseEnvironment
from aml_defense_gym.models import AMLTaskType, SanctionsDisposition


def sanctions_action(obs) -> Dict[str, Any]:
    row = obs.records[0] if obs.records else {}
    score = float(row.get("algorithmic_score", 0.0))
    if score >= 0.85:
        disp = SanctionsDisposition.ESCALATE.value
    else:
        disp = SanctionsDisposition.CLEAR_FALSE_POSITIVE.value
    return {"task": AMLTaskType.SANCTIONS_SCREENING.value, "sanctions_disposition": disp}


def edd_action() -> Dict[str, Any]:
    blob = {
        "adverse_media_summary": "No adverse media found in synthetic watchlists (deterministic stub).",
        "beneficial_ownership": "Ownership traced to disclosed parent holding (synthetic).",
        "geography_risk": "Jurisdiction flagged high-risk; corroborated with public FATF lists.",
        "conclusion": "Retain enhanced monitoring; no immediate de-risk action required.",
    }
    return {"task": AMLTaskType.EDD_REVIEW.value, "edd_report": blob}


def tm_action(obs) -> Dict[str, Any]:
    scores: List[float] = []
    for rec in obs.records:
        illicit = int(rec.get("is_laundering", 0) or 0)
        scores.append(0.99 if illicit == 1 else 0.01)
    return {"task": AMLTaskType.TRANSACTION_MONITORING.value, "transaction_scores": scores}


def policy(obs) -> Dict[str, Any]:
    if obs.task == AMLTaskType.SANCTIONS_SCREENING:
        if not obs.records:
            return {"task": obs.task.value, "sanctions_disposition": SanctionsDisposition.REQUEST_INFO.value}
        return sanctions_action(obs)
    if obs.task == AMLTaskType.EDD_REVIEW:
        return edd_action()
    return tm_action(obs)


def run_episode(task: AMLTaskType, seed: int = 42) -> Dict[str, Any]:
    env = AMLDefenseEnvironment()
    obs = env.reset(seed=seed, task=task)
    rewards: List[float] = []
    steps = 0
    while steps < 32:
        res = env.step(policy(obs))
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
        AMLTaskType.SANCTIONS_SCREENING,
        AMLTaskType.EDD_REVIEW,
        AMLTaskType.TRANSACTION_MONITORING,
    )
    rows = [run_episode(t) for t in tasks]
    mean_score = sum(r["episode_score"] for r in rows) / len(rows)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "aml-defense-gym",
        "method": "offline_oracle_policy",
        "mean_task_score": round(mean_score, 4),
        "tasks": {r["task"]: r for r in rows},
    }
    path = Path(__file__).resolve().parent.parent / "baseline_scores.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
