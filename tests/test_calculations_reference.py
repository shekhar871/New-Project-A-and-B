from __future__ import annotations

from agentguard_gym.graders import RewardConfig, cyber_minmax_bounds
from agentguard_gym.reward_math import clamp01, minmax_normalize, mttd_mttr_step_potential


def test_cyber_minmax_bounds_match_reference_doc_defaults() -> None:
    """
    Mirrors `CALCULATIONS_REFERENCE.md`:
      worst = w_fn + action_fee + min(fn_extra_prompt, fn_extra_ssrf, fn_extra_memory)
      best  = w_tp + mttd_scale + mttr_scale + action_fee
    """
    cfg = RewardConfig()
    worst, best = cyber_minmax_bounds(cfg)
    assert worst == cfg.w_fn + cfg.action_fee + min(cfg.fn_extra_prompt, cfg.fn_extra_ssrf, cfg.fn_extra_memory)
    assert best == cfg.w_tp + cfg.mttd_scale + cfg.mttr_scale + cfg.action_fee
    assert worst < best


def test_minmax_normalize_maps_bounds_to_unit_interval() -> None:
    cfg = RewardConfig()
    worst, best = cyber_minmax_bounds(cfg)
    assert minmax_normalize(worst, worst, best) == 0.0
    assert minmax_normalize(best, worst, best) == 1.0
    assert 0.0 <= minmax_normalize((worst + best) / 2.0, worst, best) <= 1.0


def test_clamp01_hard_clamps() -> None:
    assert clamp01(-1.0) == 0.0
    assert clamp01(2.0) == 1.0
    assert clamp01(0.4) == 0.4


def test_mttd_mttr_potential_is_bounded_and_monotone_in_delay() -> None:
    # Earlier detection/remediation should never be worse than later ones.
    early = mttd_mttr_step_potential(
        step_idx=0,
        detected_step=0,
        remediated_step=0,
        mttd_scale=0.15,
        mttr_scale=0.1,
    )
    late = mttd_mttr_step_potential(
        step_idx=10,
        detected_step=5,
        remediated_step=9,
        mttd_scale=0.15,
        mttr_scale=0.1,
    )
    assert early >= late
    assert 0.0 <= late <= 0.15 + 0.1

