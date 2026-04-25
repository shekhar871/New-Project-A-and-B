"""GRPO rewards use SHA registry, not fragile [GT:] substrings in the prompt."""

from agentguard_gym.training.reward_fns import (
    clear_training_prompt_metadata,
    defender_reward_fn,
    register_training_prompt_metadata,
    reset_outcome_tally,
)


def test_reward_uses_registry_not_tags() -> None:
    reset_outcome_tally()
    p = "=== ALERT ===\nSome event\n\nReturn JSON.\nTASK: prompt_injection\n"
    register_training_prompt_metadata(prompt=p, is_malicious=True, task="prompt_injection")
    comp = '{"decision":"block","confidence":0.9}'
    r = defender_reward_fn([p], [comp])[0]
    assert r > 0.4
    # Same completion, wrong tag if we relied on string — we did not put [GT:malicious] in p
    clear_training_prompt_metadata()
    reset_outcome_tally()
