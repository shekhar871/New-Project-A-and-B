from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CyberTaskType(str, Enum):
    """Three real SOC-style tasks mapped to OWASP Agentic risk themes (2026 framing)."""

    PROMPT_INJECTION = "prompt_injection"  # ASI01 — easiest: single text decision
    TOOL_MISUSE_SSRF = "tool_misuse_ssrf"  # ASI02 / ASI05 — medium: URL + tool trace
    MEMORY_POISONING_PRIVILEGE = "memory_poisoning_privilege"  # ASI03 / ASI06 — hardest: multi-signal memory


class TaskDifficulty(str, Enum):
    """Human-readable ramp so reviewers see easy → medium → hard at a glance."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def difficulty_for_task(task: CyberTaskType) -> TaskDifficulty:
    if task == CyberTaskType.PROMPT_INJECTION:
        return TaskDifficulty.EASY
    if task == CyberTaskType.TOOL_MISUSE_SSRF:
        return TaskDifficulty.MEDIUM
    return TaskDifficulty.HARD


class DefenseActionType(str, Enum):
    ALLOW = "allow"
    SANITIZE = "sanitize"
    BLOCK = "block"
    QUARANTINE_MEMORY = "quarantine_memory"
    CLEAR_EXPOSED_SECRETS = "clear_exposed_secrets"
    AUDIT_TOOL_CHAIN = "audit_tool_chain"


class AgentGuardAction(BaseModel):
    """What the defensive agent wants to do this turn (typed so bad JSON fails loud and early)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    defense: DefenseActionType
    rationale: str = Field(default="", max_length=4096)
    target_id: Optional[str] = Field(
        default=None,
        description="Optional handle for the artifact under review (prompt id, URL id, memory slot).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentGuardReward(BaseModel):
    """
    Scalar the hackathon autograder cares about, plus the raw utility for transparency.

    `value` is always clipped to [0, 1]; `utility_raw` is the pre-normalized signal we used to
    derive that score (handy when you are debugging reward shaping in a notebook).
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0, description="Normalized score used in stdout logs.")
    utility_raw: float = Field(description="Unscaled utility before min–max normalization.")
    outcome: Optional[Literal["tp", "fp", "fn", "tn"]] = Field(
        default=None,
        description="Confusion-matrix bucket for this step, when it applies.",
    )
    partial_credit: bool = Field(
        default=False,
        description="True when the action was directionally right but not the gold-standard response.",
    )


class AgentGuardObservation(BaseModel):
    """Everything the policy is allowed to see this step (no hidden labels sneaked in here)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: CyberTaskType
    task_difficulty: TaskDifficulty
    step_index: int = Field(ge=0)
    episode_id: str
    narrative: str = Field(description="Natural-language context for the policy.")
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured handles (prompt text, URLs, memory previews, etc.).",
    )
    tool_trace: List[str] = Field(
        default_factory=list,
        description="Synthetic tool calls the blue team can monitor.",
    )
    validation_error: Optional[str] = Field(
        default=None,
        description="If the last action broke the schema, we surface it here as readable text.",
    )


class AgentGuardState(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    episode_id: str
    step_count: int = Field(ge=0)
    task: CyberTaskType
    task_difficulty: TaskDifficulty
    seed: Optional[int] = None
    mttd_steps: Optional[int] = Field(
        default=None,
        description="First step index where a real threat was handled correctly (proxy for MTTD).",
    )
    mttr_steps: Optional[int] = Field(
        default=None,
        description="Steps from detection proxy to remediation proxy (rough MTTR stand-in).",
    )


class StepResult(BaseModel):
    """What `step()` returns: new view of the world, normalized reward, terminal flag, debug info."""

    model_config = ConfigDict(extra="forbid")

    observation: AgentGuardObservation
    reward: AgentGuardReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
