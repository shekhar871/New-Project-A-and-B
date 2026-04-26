import json
from pathlib import Path

import pytest

import agentguard_gym.grandfinals.adversarial_loop as adv_loop
from agentguard_gym.grandfinals.adversarial_loop import AdversarialSystem
from agentguard_gym.models import CyberAdversarialTaskType


@pytest.fixture(scope="module")
def system() -> AdversarialSystem:
    root = Path(__file__).resolve().parent.parent
    seeds = json.loads((root / "data" / "attack_corpus.json").read_text(encoding="utf-8"))
    return AdversarialSystem.from_seed_attacks(seeds, data_dir=root / "data")


def test_benign_outcome_types(system: AdversarialSystem, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force benign branch
    monkeypatch.setattr(adv_loop.random, "random", lambda: 0.0)
    results = [system.run_episode(task=CyberAdversarialTaskType.PROMPT_INJECTION, step_idx=i) for i in range(30)]
    for r in results:
        assert r.is_malicious_scenario is False
        assert r.outcome in {"TN", "FP"}, f"Benign routed to malicious outcome: {r.outcome}"


def test_benign_reward_signs(system: AdversarialSystem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adv_loop.random, "random", lambda: 0.0)
    results = [system.run_episode(task=CyberAdversarialTaskType.PROMPT_INJECTION, step_idx=i) for i in range(30)]
    for r in results:
        if r.outcome == "TN":
            assert r.reward_defender > 0.5, f"TN has unexpectedly low normalized R_D={r.reward_defender:.4f}"
        elif r.outcome == "FP":
            assert r.reward_defender < 0.5, f"FP has unexpectedly high normalized R_D={r.reward_defender:.4f}"


def test_elo_sign_on_benign(system: AdversarialSystem, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run one forced benign episode and ensure defender_won is aligned with TN.
    monkeypatch.setattr(adv_loop.random, "random", lambda: 0.0)
    r = system.run_episode(task=CyberAdversarialTaskType.PROMPT_INJECTION, step_idx=0)
    assert r.defender_won is (r.outcome == "TN")

