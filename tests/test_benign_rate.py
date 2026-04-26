def test_benign_rate() -> None:
    from agentguard_gym.grandfinals.adversarial_loop import BENIGN_EPISODE_PROB

    import random

    rng = random.Random(123)
    hits = [rng.random() < BENIGN_EPISODE_PROB for _ in range(500)]
    rate = hits.count(True) / 500
    assert 0.22 <= rate <= 0.38, f"Benign rate {rate:.1%} out of band"


def test_benign_prob_value() -> None:
    from agentguard_gym.grandfinals.adversarial_loop import BENIGN_EPISODE_PROB

    assert BENIGN_EPISODE_PROB == 0.30

