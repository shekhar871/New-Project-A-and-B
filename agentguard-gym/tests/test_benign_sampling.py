"""Tests that BENIGN_EPISODE_PROB fires at the declared rate."""

import random

from agentguard_gym.grandfinals.adversarial_loop import BENIGN_EPISODE_PROB


def test_benign_prob_constant():
    """Constant must be exactly 0.30."""
    assert BENIGN_EPISODE_PROB == 0.30, (
        f"BENIGN_EPISODE_PROB is {BENIGN_EPISODE_PROB}, expected 0.30"
    )


def test_benign_sampling_rate():
    """random.random() < BENIGN_EPISODE_PROB fires at 20–40% over 500 trials."""
    random.seed(42)
    benign_count = sum(
        1 for _ in range(500) if random.random() < BENIGN_EPISODE_PROB
    )
    rate = benign_count / 500
    assert 0.20 <= rate <= 0.40, (
        f"Benign sampling rate {rate:.1%} outside expected 20–40% band"
    )

