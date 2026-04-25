"""Benign (TN/FP) path for AdversarialSystem — avoids always-block collapse."""

import json
from pathlib import Path

import pytest

import agentguard_gym.grandfinals.adversarial_loop as adv_loop
from agentguard_gym.models import CyberAdversarialTaskType
from agentguard_gym.grandfinals.adversarial_loop import AdversarialSystem


@pytest.fixture
def system() -> AdversarialSystem:
    root = Path(__file__).resolve().parent.parent
    seeds = json.loads((root / "data" / "attack_corpus.json").read_text(encoding="utf-8"))
    return AdversarialSystem.from_seed_attacks(
        seeds,
        data_dir=root / "data",
    )


def test_benign_episode_sets_flag_and_tn_or_fp(system: AdversarialSystem, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the ~30% benign branch
    monkeypatch.setattr(adv_loop.random, "random", lambda: 0.1)
    r = system.run_episode(task=CyberAdversarialTaskType.PROMPT_INJECTION, step_idx=0)
    assert r.is_malicious_scenario is False
    assert r.outcome in ("TN", "FP")
    assert r.defender_won is (r.outcome == "TN")
