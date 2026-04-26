from __future__ import annotations

import random

from agentguard_gym.graders import (
    RewardConfig,
    cyber_minmax_bounds,
    grade_memory_poisoning,
    grade_prompt_injection,
    grade_ssrf_tool_misuse,
)
from agentguard_gym.models import AgentGuardAction, DefenseActionType
from agentguard_gym.reward_math import minmax_normalize, mttd_mttr_step_potential


def _mk_action(defense: DefenseActionType) -> AgentGuardAction:
    return AgentGuardAction(defense=defense, rationale="test")


def test_reward_ordering_prompt_injection_tp_tn_fp_fn() -> None:
    cfg = RewardConfig()
    worst, best = cyber_minmax_bounds(cfg)

    # Fix timing metadata so we compare apples-to-apples.
    step_idx = 1
    detected_step = 1
    remediated_step = 1

    u_tp, out_tp, *_ = grade_prompt_injection(
        True, _mk_action(DefenseActionType.BLOCK), cfg, step_idx=step_idx, detected_step=detected_step, remediated_step=remediated_step
    )
    u_fn, out_fn, *_ = grade_prompt_injection(
        True, _mk_action(DefenseActionType.ALLOW), cfg, step_idx=step_idx, detected_step=detected_step, remediated_step=remediated_step
    )
    u_tn, out_tn, *_ = grade_prompt_injection(
        False, _mk_action(DefenseActionType.ALLOW), cfg, step_idx=step_idx, detected_step=detected_step, remediated_step=remediated_step
    )
    u_fp, out_fp, *_ = grade_prompt_injection(
        False, _mk_action(DefenseActionType.BLOCK), cfg, step_idx=step_idx, detected_step=detected_step, remediated_step=remediated_step
    )

    assert out_tp == "tp"
    assert out_tn == "tn"
    assert out_fp == "fp"
    assert out_fn == "fn"

    # Core invariant: catching a real attack should be best; missing it should be worst.
    assert u_tp > u_tn > u_fp > u_fn

    # Normalization must preserve ordering and stay in [0,1].
    r_tp = minmax_normalize(u_tp, worst, best)
    r_tn = minmax_normalize(u_tn, worst, best)
    r_fp = minmax_normalize(u_fp, worst, best)
    r_fn = minmax_normalize(u_fn, worst, best)
    assert 0.0 <= r_fn <= 1.0
    assert 0.0 <= r_fp <= 1.0
    assert 0.0 <= r_tn <= 1.0
    assert 0.0 <= r_tp <= 1.0
    assert r_tp > r_tn > r_fp > r_fn


def test_reward_ordering_ssrf_tp_tn_fp_fn_and_partial_credit() -> None:
    cfg = RewardConfig()

    # Threat case: block is gold TP; audit is TP partial; allow is FN.
    u_tp_block, out1, _, partial1 = grade_ssrf_tool_misuse(
        True, _mk_action(DefenseActionType.BLOCK), cfg, step_idx=0, detected_step=0, remediated_step=0
    )
    u_tp_audit, out2, _, partial2 = grade_ssrf_tool_misuse(
        True, _mk_action(DefenseActionType.AUDIT_TOOL_CHAIN), cfg, step_idx=0, detected_step=0, remediated_step=0
    )
    u_fn, out3, *_ = grade_ssrf_tool_misuse(
        True, _mk_action(DefenseActionType.ALLOW), cfg, step_idx=0, detected_step=0, remediated_step=0
    )

    assert out1 == "tp" and partial1 is False
    assert out2 == "tp" and partial2 is True
    assert out3 == "fn"
    assert u_tp_block > u_tp_audit > u_fn

    # Benign case: allow is TN; block is FP; audit is TN (safe inspection).
    u_tn_allow, out4, *_ = grade_ssrf_tool_misuse(
        False, _mk_action(DefenseActionType.ALLOW), cfg, step_idx=0, detected_step=None, remediated_step=None
    )
    u_fp_block, out5, *_ = grade_ssrf_tool_misuse(
        False, _mk_action(DefenseActionType.BLOCK), cfg, step_idx=0, detected_step=None, remediated_step=None
    )
    u_tn_audit, out6, *_ = grade_ssrf_tool_misuse(
        False, _mk_action(DefenseActionType.AUDIT_TOOL_CHAIN), cfg, step_idx=0, detected_step=None, remediated_step=None
    )
    assert out4 == "tn" and out6 == "tn" and out5 == "fp"
    assert u_tn_allow > u_fp_block
    assert u_tn_audit > u_fp_block


def test_reward_ordering_memory_tp_tn_fp_fn_and_partial_credit() -> None:
    cfg = RewardConfig()

    # Threat case: allow is FN; quarantine/clear are gold TP depending on threat; block is partial TP.
    u_fn, out1, *_ = grade_memory_poisoning(
        True, False, _mk_action(DefenseActionType.ALLOW), cfg, step_idx=0, detected_step=0, remediated_step=0
    )
    u_tp_gold, out2, _, partial2 = grade_memory_poisoning(
        True, False, _mk_action(DefenseActionType.QUARANTINE_MEMORY), cfg, step_idx=0, detected_step=0, remediated_step=0
    )
    u_tp_partial, out3, _, partial3 = grade_memory_poisoning(
        True, False, _mk_action(DefenseActionType.BLOCK), cfg, step_idx=0, detected_step=0, remediated_step=0
    )
    assert out1 == "fn"
    assert out2 == "tp" and partial2 is False
    assert out3 == "tp" and partial3 is True
    assert u_tp_gold > u_tp_partial > u_fn

    # Benign case: allow TN; block/quarantine FP.
    u_tn, out4, *_ = grade_memory_poisoning(
        False, False, _mk_action(DefenseActionType.ALLOW), cfg, step_idx=0, detected_step=None, remediated_step=None
    )
    u_fp, out5, *_ = grade_memory_poisoning(
        False, False, _mk_action(DefenseActionType.BLOCK), cfg, step_idx=0, detected_step=None, remediated_step=None
    )
    assert out4 == "tn" and out5 == "fp"
    assert u_tn > u_fp


def test_time_bonus_is_bounded_under_random_timings() -> None:
    rng = random.Random(0)
    for _ in range(500):
        det = rng.randint(0, 20)
        rem = rng.randint(0, 20)
        bonus = mttd_mttr_step_potential(
            step_idx=rng.randint(0, 50),
            detected_step=det,
            remediated_step=rem,
            mttd_scale=0.15,
            mttr_scale=0.1,
        )
        assert 0.0 <= bonus <= 0.25

