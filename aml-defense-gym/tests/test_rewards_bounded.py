from aml_defense_gym.environment import AMLDefenseEnvironment
from aml_defense_gym.models import AMLTaskType, SanctionsDisposition


def test_sanctions_episode_rewards_are_normalized() -> None:
    env = AMLDefenseEnvironment()
    env.reset(seed=1, task=AMLTaskType.SANCTIONS_SCREENING)
    while True:
        res = env.step(
            {
                "task": AMLTaskType.SANCTIONS_SCREENING.value,
                "sanctions_disposition": SanctionsDisposition.ESCALATE.value,
            }
        )
        assert 0.0 <= res.reward.value <= 1.0
        if res.done:
            break


def test_transaction_monitoring_uses_unit_interval() -> None:
    env = AMLDefenseEnvironment()
    obs = env.reset(seed=2, task=AMLTaskType.TRANSACTION_MONITORING)
    n = len(obs.records)
    scores = [i / max(1, n) for i in range(n)]
    res = env.step(
        {
            "task": AMLTaskType.TRANSACTION_MONITORING.value,
            "transaction_scores": scores,
        }
    )
    assert 0.0 <= res.reward.value <= 1.0
