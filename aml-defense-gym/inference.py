"""
Baseline inference for AML-DefenseGym with the exact stdout contract the hackathon autograder expects.

The LLM must emit JSON that matches `AMLAction` — we keep temperatures at zero so repeated runs
on the same endpoint do not jitter wildly (still subject to provider-side nondeterminism).
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from aml_defense_gym.environment import AMLDefenseEnvironment
from aml_defense_gym.models import AMLTaskType, SanctionsDisposition

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("AML_DEFENSE_BENCHMARK", "aml-defense-gym")

MAX_STEPS = 16
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.65

TASK_ORDER = (
    AMLTaskType.SANCTIONS_SCREENING,
    AMLTaskType.EDD_REVIEW,
    AMLTaskType.TRANSACTION_MONITORING,
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = "null" if error is None else error.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    joined = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={joined}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AML compliance agent. Output ONE JSON object only (no markdown), shaped for the task:
    - Sanctions: {"task":"sanctions_screening","sanctions_disposition":"clear_false_positive|escalate|request_info"}
    - EDD: {"task":"edd_review","edd_report":{
        "adverse_media_summary":"...",
        "beneficial_ownership":"...",
        "geography_risk":"...",
        "conclusion":"..."
      }}
    - Transactions: {"task":"transaction_monitoring","transaction_scores":[0.0-1.0, ...]}
      Length of transaction_scores MUST equal number of records shown.
    Include only keys needed for the current task.
    """
).strip()


@dataclass
class AMLSession:
    env: AMLDefenseEnvironment

    def close(self) -> None:
        return None


def _client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN / OPENAI_API_KEY missing — export it before running inference.")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _llm_action(client: OpenAI, observation_json: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    user = json.dumps(observation_json, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        payload = json.loads(text)
        action = dict(payload)
        if "task" in action and isinstance(action["task"], str):
            pass
        return json.dumps(action, ensure_ascii=False), action, None
    except Exception as exc:  # noqa: BLE001
        fallback = {
            "task": observation_json.get("task", AMLTaskType.SANCTIONS_SCREENING.value),
            "sanctions_disposition": SanctionsDisposition.REQUEST_INFO.value,
            "rationale": f"fallback:{exc}",
        }
        return json.dumps(fallback), fallback, str(exc)


def run_task(task: AMLTaskType, client: OpenAI) -> Tuple[bool, int, float, List[float]]:
    session = AMLSession(env=AMLDefenseEnvironment())
    rewards: List[float] = []
    steps = 0
    obs = session.env.reset(seed=42, task=task)
    ended = False

    log_start(task.value, BENCHMARK, MODEL_NAME)

    try:
        while steps < MAX_STEPS:
            obs_dict = obs.model_dump(mode="json")
            action_str, action_dict, llm_err = _llm_action(client, obs_dict)
            result = session.env.step(action_dict)
            steps += 1
            r_val = float(result.reward.value)
            rewards.append(r_val)
            err = llm_err or result.observation.validation_error or result.info.get("detail")
            if isinstance(err, str) and err.strip() == "":
                err = None
            log_step(steps, action_str, r_val, result.done, err)
            obs = result.observation
            if result.done:
                break

        score = sum(rewards) / max(1, len(rewards))
        success = score >= SUCCESS_THRESHOLD
        log_end(success, steps, score, rewards)
        ended = True
        return success, steps, score, rewards
    finally:
        if not ended:
            log_end(False, steps, 0.0, rewards or [0.0])
        session.close()


def main() -> None:
    client = _client()
    aggregate: List[float] = []
    for task in TASK_ORDER:
        _success, _steps, score, _rewards = run_task(task, client)
        aggregate.append(score)
    overall = sum(aggregate) / len(aggregate)
    print(f"[SUMMARY] benchmark={BENCHMARK} mean_task_score={overall:.2f}", flush=True)


if __name__ == "__main__":
    main()
