"""
Small numerics helpers for AML grading.

The competition wants every scalar reward in [0, 1]. Real compliance models usually work on
asymmetric cost matrices (Elkan, "The Foundations of Cost-Sensitive Learning", 2001) where
missing a launderer is vastly more expensive than annoying a good customer. We keep that *relative*
ordering in the raw utility, then squash into [0, 1] with a task-specific min–max map so judges
can still read the economics in `utility_raw` inside the Pydantic payload.

For streaming monitoring we borrow the spirit of recall@k + alert-workload penalties discussed in
industry transaction-monitoring write-ups (e.g., Silent Eight / Verafin public commentary on alert
fatigue). Here it is a lightweight convex blend so partial recall still yields partial credit.
"""

from __future__ import annotations


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def minmax_normalize(value: float, worst: float, best: float) -> float:
    if best <= worst:
        return clamp01(value)
    return clamp01((value - worst) / (best - worst))
