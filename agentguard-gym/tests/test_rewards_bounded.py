from agentguard_gym.environment import AgentGuardEnvironment
from agentguard_gym.models import CyberTaskType, DefenseActionType


def test_full_episode_rewards_stay_in_unit_interval() -> None:
    env = AgentGuardEnvironment()
    env.reset(seed=7, task=CyberTaskType.PROMPT_INJECTION)
    while True:
        res = env.step({"defense": DefenseActionType.ALLOW.value, "rationale": "allow benign traffic"})
        assert 0.0 <= res.reward.value <= 1.0
        if res.done:
            break
