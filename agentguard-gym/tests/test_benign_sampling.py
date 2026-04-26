"""Verifies BENIGN_EPISODE_PROB fires at the declared ~30% rate."""
import random
import pytest
from agentguard_gym.grandfinals.adversarial_loop import BENIGN_EPISODE_PROB


def test_benign_prob_constant():
    assert BENIGN_EPISODE_PROB == 0.30, (
        f"BENIGN_EPISODE_PROB is {BENIGN_EPISODE_PROB}, must be 0.30"
    )


def test_benign_sampling_rate():
    random.seed(42)
    hits = sum(1 for _ in range(500) if random.random() < BENIGN_EPISODE_PROB)
    rate = hits / 500
    assert 0.20 <= rate <= 0.40, (
        f"Benign rate {rate:.1%} outside 20–40% band — "
        f"check BENIGN_EPISODE_PROB and the sampling branch in adversarial_loop.py"
    )

