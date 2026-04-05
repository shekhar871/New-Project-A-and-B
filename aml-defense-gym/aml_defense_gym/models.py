from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AMLTaskType(str, Enum):
    SANCTIONS_SCREENING = "sanctions_screening"
    EDD_REVIEW = "edd_review"
    TRANSACTION_MONITORING = "transaction_monitoring"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def difficulty_for_task(task: AMLTaskType) -> TaskDifficulty:
    if task == AMLTaskType.SANCTIONS_SCREENING:
        return TaskDifficulty.EASY
    if task == AMLTaskType.EDD_REVIEW:
        return TaskDifficulty.MEDIUM
    return TaskDifficulty.HARD


class SanctionsDisposition(str, Enum):
    CLEAR_FALSE_POSITIVE = "clear_false_positive"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"


class AMLAction(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: AMLTaskType
    sanctions_disposition: Optional[SanctionsDisposition] = None
    edd_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured narrative + evidence keys for EDD grading.",
    )
    transaction_scores: Optional[List[float]] = Field(
        default=None,
        description="Risk scores 0..1 for each transaction in the current batch (monitoring task).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AMLReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    utility_raw: float
    outcome: Optional[str] = None
    partial_credit: bool = False


class AMLObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: AMLTaskType
    task_difficulty: TaskDifficulty
    step_index: int = Field(ge=0)
    episode_id: str
    narrative: str
    records: List[Dict[str, Any]] = Field(default_factory=list)
    validation_error: Optional[str] = None


class AMLState(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    episode_id: str
    step_count: int = Field(ge=0)
    task: AMLTaskType
    task_difficulty: TaskDifficulty
    seed: Optional[int] = None
    window_closed: bool = Field(
        default=False,
        description="For delayed-feedback transaction monitoring windows.",
    )


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: AMLObservation
    reward: AMLReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
