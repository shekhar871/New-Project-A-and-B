from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import ValidationError

from aml_defense_gym.data_loader import default_csv_path, iter_transaction_chunks
from aml_defense_gym.graders import (
    AMLRewardConfig,
    edd_bounds,
    grade_edd,
    grade_sanctions,
    grade_transaction_batch,
    sanctions_bounds,
)
from aml_defense_gym.models import (
    AMLAction,
    AMLObservation,
    AMLReward,
    AMLState,
    AMLTaskType,
    SanctionsDisposition,
    StepResult,
    difficulty_for_task,
)
from aml_defense_gym.reward_math import minmax_normalize


class AMLDefenseEnvironment:
    """
    Synthetic AML ops room: sanctions hits, EDD memos, and a fatiguing transaction batch.

    Nothing here touches real PII — it is a sandbox for agents to practice the *shape* of the work:
    disposition lists, evidence gathering, and ranking alerts under imbalance.
    """

    def __init__(
        self,
        reward_config: Optional[AMLRewardConfig] = None,
        *,
        transaction_csv: Optional[Path] = None,
    ) -> None:
        self._rc = reward_config or AMLRewardConfig()
        self._tx_path = transaction_csv
        self._rng = random.Random()
        self._episode_id = ""
        self._task: AMLTaskType = AMLTaskType.SANCTIONS_SCREENING
        self._step = 0
        self._seed: Optional[int] = None
        self._phase = 0
        self._sanctions_queue: List[Dict[str, Any]] = []
        self._edd_profile: Dict[str, Any] = {}
        self._tx_batch: pd.DataFrame = pd.DataFrame()
        self._window_closed = False
        self._san_low, self._san_high = sanctions_bounds(self._rc)
        self._edd_low, self._edd_high = edd_bounds()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[AMLTaskType] = None,
    ) -> AMLObservation:
        self._seed = seed
        self._rng = random.Random(seed if seed is not None else random.randrange(2**31))
        self._episode_id = episode_id or str(uuid.uuid4())
        self._task = task or self._rng.choice(list(AMLTaskType))
        self._step = 0
        self._phase = 0
        self._window_closed = False

        if self._task == AMLTaskType.SANCTIONS_SCREENING:
            self._sanctions_queue = self._build_sanctions_episode()
        elif self._task == AMLTaskType.EDD_REVIEW:
            self._edd_profile = self._build_edd_profile()
        else:
            self._tx_batch = self._sample_tx_batch()

        return self._observation()

    def state(self) -> AMLState:
        return AMLState(
            episode_id=self._episode_id,
            step_count=self._step,
            task=self._task,
            task_difficulty=difficulty_for_task(self._task),
            seed=self._seed,
            window_closed=self._window_closed,
        )

    def step(self, raw_action: Any) -> StepResult:
        try:
            action = AMLAction.model_validate(raw_action)
        except ValidationError as exc:
            obs = self._observation(validation_error=str(exc))
            utility_raw = -3.0
            value = minmax_normalize(utility_raw, self._edd_low, self._edd_high)
            reward = AMLReward(value=value, utility_raw=utility_raw, outcome=None, partial_credit=False)
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                info={"error": "validation_error", "detail": str(exc)},
            )

        if action.task != self._task:
            obs = self._observation()
            utility_raw = -1.8
            value = minmax_normalize(utility_raw, self._edd_low, self._edd_high)
            reward = AMLReward(value=value, utility_raw=utility_raw, outcome=None, partial_credit=False)
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                info={"error": "task_mismatch"},
            )

        if self._task == AMLTaskType.SANCTIONS_SCREENING:
            return self._step_sanctions(action)
        if self._task == AMLTaskType.EDD_REVIEW:
            return self._step_edd(action)
        return self._step_transactions(action)

    def _build_sanctions_episode(self) -> List[Dict[str, Any]]:
        names = [
            {"name": "John Smith", "dob": "1980-01-02", "score": 0.12, "true_match": False},
            {"name": "ALEKSEI SERGEYEV", "dob": "1975-05-15", "score": 0.97, "true_match": True},
            {"name": "Maria Garcia", "dob": "1992-03-09", "score": 0.44, "true_match": False},
        ]
        self._rng.shuffle(names)
        return names

    def _build_edd_profile(self) -> Dict[str, Any]:
        return {
            "customer_id": "C-" + "".join(self._rng.choice("0123456789") for _ in range(6)),
            "risk_tier": "high",
            "jurisdiction": "offshore_shell_list",
            "notes": "Flagged for velocity and PEP proximity (synthetic).",
        }

    def _sample_tx_batch(self) -> pd.DataFrame:
        path = self._tx_path or default_csv_path()
        chunks = list(iter_transaction_chunks(path, chunksize=500))
        if not chunks:
            return pd.DataFrame()
        chunk = chunks[0]
        n = min(5, len(chunk))
        return chunk.sample(n=n, random_state=self._seed).reset_index(drop=True)

    def _observation(self, validation_error: Optional[str] = None) -> AMLObservation:
        diff = difficulty_for_task(self._task)
        if self._task == AMLTaskType.SANCTIONS_SCREENING:
            if self._phase >= len(self._sanctions_queue):
                narrative = "Sanctions screening episode complete."
                records: List[Dict[str, Any]] = []
            else:
                row = self._sanctions_queue[self._phase]
                narrative = (
                    "Disposition this screening hit. Cross-check secondary identifiers "
                    "against watchlist ground truth (hidden from agent)."
                )
                records = [
                    {
                        "candidate_name": row["name"],
                        "dob": row["dob"],
                        "algorithmic_score": row["score"],
                    }
                ]
            return AMLObservation(
                task=self._task,
                task_difficulty=diff,
                step_index=self._step,
                episode_id=self._episode_id,
                narrative=narrative,
                records=records,
                validation_error=validation_error,
            )

        if self._task == AMLTaskType.EDD_REVIEW:
            narrative = (
                "Produce an Enhanced Due Diligence memo with mandatory regulatory fields "
                "(adverse media, beneficial ownership, geography, conclusion)."
            )
            records = [self._edd_profile] if self._phase == 0 else []
            return AMLObservation(
                task=self._task,
                task_difficulty=diff,
                step_index=self._step,
                episode_id=self._episode_id,
                narrative=narrative,
                records=records,
                validation_error=validation_error,
            )

        narrative = (
            "Streaming batch: assign suspicion scores [0,1] aligned with laundering labels "
            "(feedback delayed until window closes — simulated in one shot here)."
        )
        records = self._tx_batch.to_dict(orient="records") if not self._tx_batch.empty else []
        return AMLObservation(
            task=self._task,
            task_difficulty=diff,
            step_index=self._step,
            episode_id=self._episode_id,
            narrative=narrative,
            records=records,
            validation_error=validation_error,
        )

    def _step_sanctions(self, action: AMLAction) -> StepResult:
        if self._phase >= len(self._sanctions_queue):
            obs = self._observation()
            reward = AMLReward(value=0.0, utility_raw=0.0, outcome=None, partial_credit=False)
            return StepResult(observation=obs, reward=reward, done=True, info={"reason": "complete"})

        if action.sanctions_disposition is None:
            obs = self._observation()
            utility_raw = -2.0
            value = minmax_normalize(utility_raw, self._san_low, self._san_high)
            reward = AMLReward(value=value, utility_raw=utility_raw, outcome=None, partial_credit=False)
            return StepResult(observation=obs, reward=reward, done=False, info={"error": "missing_disposition"})

        row = self._sanctions_queue[self._phase]
        utility_raw, outcome, partial = grade_sanctions(row["true_match"], action.sanctions_disposition, self._rc)
        value = minmax_normalize(utility_raw, self._san_low, self._san_high)
        self._phase += 1
        self._step += 1
        done = self._phase >= len(self._sanctions_queue)
        obs = self._observation()
        reward = AMLReward(value=value, utility_raw=utility_raw, outcome=outcome, partial_credit=partial)
        return StepResult(observation=obs, reward=reward, done=done, info={"outcome": outcome, "partial_credit": partial})

    def _step_edd(self, action: AMLAction) -> StepResult:
        utility_raw, meta = grade_edd(action.edd_report, self._rc)
        value = minmax_normalize(utility_raw, self._edd_low, self._edd_high)
        self._step += 1
        self._phase = 1
        done = bool(meta.get("complete", False))
        obs = self._observation()
        partial = not done and utility_raw > self._edd_low
        reward = AMLReward(
            value=value,
            utility_raw=utility_raw,
            outcome="tp" if done else None,
            partial_credit=partial,
        )
        return StepResult(observation=obs, reward=reward, done=done, info=meta)

    def _step_transactions(self, action: AMLAction) -> StepResult:
        if self._tx_batch.empty:
            obs = self._observation()
            reward = AMLReward(value=0.0, utility_raw=0.0, outcome=None, partial_credit=False)
            return StepResult(observation=obs, reward=reward, done=True, info={"error": "no_data"})

        scores = action.transaction_scores
        if scores is None or len(scores) != len(self._tx_batch):
            obs = self._observation()
            utility_raw = 0.05
            reward = AMLReward(value=utility_raw, utility_raw=utility_raw, outcome=None, partial_credit=True)
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                info={"error": "scores_required", "expected": len(self._tx_batch)},
            )

        labels_col = "is_laundering"
        if labels_col not in self._tx_batch.columns:
            obs = self._observation()
            reward = AMLReward(value=0.0, utility_raw=0.0, outcome=None, partial_credit=False)
            return StepResult(observation=obs, reward=reward, done=True, info={"error": "missing_labels"})

        labels = [int(x) for x in self._tx_batch[labels_col].tolist()]
        utility_raw, meta = grade_transaction_batch(labels, scores, self._rc)
        self._step += 1
        self._window_closed = True
        obs = self._observation()
        recall = float(meta.get("recall_at_k", 0.0))
        reward = AMLReward(
            value=utility_raw,
            utility_raw=utility_raw,
            outcome="tp" if recall >= 0.999 else "fn",
            partial_credit=0.0 < recall < 1.0,
        )
        return StepResult(observation=obs, reward=reward, done=True, info=meta)
