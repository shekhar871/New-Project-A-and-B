from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from aml_defense_gym.models import SanctionsDisposition
from aml_defense_gym.reward_math import clamp01

Outcome = Literal["tp", "fp", "fn", "tn"]


@dataclass
class AMLRewardConfig:
    """
    Raw utilities before normalization. The asymmetry mirrors published AML RL discussions
    (e.g., MDPI Appl. Sci. 2026, fraud/AML RL framing): missing illicit activity should hurt far
    more than over-scratching legitimate customers.
    """

    illicit_tp: float = 5.0
    illicit_fn: float = -25.0
    legit_tn: float = 0.5
    legit_fp: float = -2.0
    action_fee: float = -0.01


def sanctions_bounds(cfg: AMLRewardConfig) -> Tuple[float, float]:
    """Worst case ≈ false negative, best case ≈ true positive (both include the tiny fee)."""
    return cfg.illicit_fn + cfg.action_fee, cfg.illicit_tp + cfg.action_fee


def grade_sanctions(
    true_match: bool,
    disposition: SanctionsDisposition,
    cfg: AMLRewardConfig,
) -> Tuple[float, Outcome, bool]:
    escalated = disposition == SanctionsDisposition.ESCALATE
    cleared = disposition == SanctionsDisposition.CLEAR_FALSE_POSITIVE

    if true_match:
        if escalated:
            return cfg.illicit_tp + cfg.action_fee, "tp", False
        if cleared:
            return cfg.illicit_fn + cfg.action_fee, "fn", False
        if disposition == SanctionsDisposition.REQUEST_INFO:
            return cfg.illicit_fn * 0.35 + cfg.action_fee, "fn", True
        return cfg.illicit_fn * 0.3 + cfg.action_fee, "fn", False
    else:
        if cleared:
            return cfg.legit_tn + cfg.action_fee, "tn", False
        if escalated:
            return cfg.legit_fp + cfg.action_fee, "fp", False
        if disposition == SanctionsDisposition.REQUEST_INFO:
            return cfg.legit_tn * 0.7 + cfg.action_fee, "tn", True
        return cfg.legit_tn * 0.5 + cfg.action_fee, "tn", False


_REQUIRED_EDD_KEYS = (
    "adverse_media_summary",
    "beneficial_ownership",
    "geography_risk",
    "conclusion",
)


def grade_edd(report: Optional[Dict[str, Any]], cfg: AMLRewardConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Regex/structure style check inspired by the hackathon brief: regulators want *evidence*, not vibes.

    We score depth coarsely via string length — not because it is perfect, but because it is
    deterministic and cheap to grade at scale (you can swap in spaCy/NER later without touching the API).
    """
    _ = cfg
    if not report:
        return -5.0, {"complete": False, "missing": list(_REQUIRED_EDD_KEYS)}
    missing = [k for k in _REQUIRED_EDD_KEYS if k not in report or not str(report.get(k, "")).strip()]
    if missing:
        return -2.5, {"complete": False, "missing": missing}
    lens = [len(str(report[k])) for k in _REQUIRED_EDD_KEYS]
    if min(lens) < 12:
        return -1.0, {"complete": False, "reason": "insufficient_depth"}
    return 2.0, {"complete": True}


def edd_bounds() -> Tuple[float, float]:
    return -5.0, 2.0


def grade_transaction_batch(
    labels: List[int],
    scores: List[float],
    cfg: AMLRewardConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Already returns a [0, 1] style utility: recall captures illicit capture, fp_pressure models alert fatigue.

    Weights (0.65 / 0.35) are a transparent knob — bump recall weight if your institution cares more
    about misses than analyst queue length.
    """
    _ = cfg
    if len(labels) != len(scores):
        return 0.0, {"error": "length_mismatch"}

    n = len(labels)
    illicit_idx = [i for i, y in enumerate(labels) if y == 1]
    if not illicit_idx:
        return 1.0, {"illicit_count": 0}

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    # Evaluation budget k must be able to cover all positives in the batch, else max recall < 1
    # even for a perfect ranker (see verification PDF §5.4). Cap k at batch size n only.
    k = min(n, max(1, len(illicit_idx)))
    topk = set(ranked[:k])
    caught = sum(1 for i in illicit_idx if i in topk)
    recall = caught / len(illicit_idx)

    fp_pressure = sum(1 for i in topk if labels[i] == 0) / float(k)

    utility01 = clamp01(0.65 * recall + 0.35 * (1.0 - fp_pressure))
    return utility01, {"recall_at_k": recall, "k": k, "fp_pressure": fp_pressure}
