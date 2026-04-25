# scripts/eval_before_after.py
# Generalization score on the committed holdout set ONLY (data/holdout_attacks.json).

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentguard_gym.models import CyberAdversarialTaskType, AttackerAction
from agentguard_gym.grandfinals.observation_builder import build_defender_prompt

HOLDOUT_PATH = ROOT / "data" / "holdout_attacks.json"
TRAINING_MARKER = ROOT / "data" / "training_started_at"


def _assert_holdout_lock() -> None:
    if not HOLDOUT_PATH.is_file():
        raise SystemExit(f"Missing holdout file: {HOLDOUT_PATH}")
    if TRAINING_MARKER.is_file():
        if HOLDOUT_PATH.stat().st_mtime >= TRAINING_MARKER.stat().st_mtime:
            raise SystemExit(
                f"Holdout {HOLDOUT_PATH} must be older than {TRAINING_MARKER} "
                "(commit holdout before running training)."
            )


def load_holdout_attacks() -> list:
    data = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    attacks = data.get("attacks")
    if not attacks or not isinstance(attacks, list):
        raise SystemExit("holdout_attacks.json must contain a non-empty 'attacks' array.")
    if len(attacks) != 15:
        raise SystemExit(f"Expected exactly 15 holdout attacks, found {len(attacks)}.")
    return attacks


def evaluate_on_held_out(model_fn, label: str) -> dict:
    """
    model_fn: callable(prompt: str) -> str (defender decision text)
    """
    _assert_holdout_lock()
    held = load_holdout_attacks()
    results: dict = {"total": 0, "correct": 0, "by_task": {}}

    for item in held:
        task = CyberAdversarialTaskType(item["task"])
        AttackerAction(
            attack_text=item["text"],
            attack_type=item.get("type", "holdout"),
            difficulty_score=0.7,
            target_task=task,
        )

        prompt = build_defender_prompt(narrative=item["text"], task=task.value, is_malicious=True)
        response = model_fn(prompt)

        match = re.search(r'"decision"\s*:\s*"(\w+)"', response)
        decision = match.group(1).lower() if match else "allow"

        correct = decision == "block"
        results["total"] += 1
        results["correct"] += int(correct)
        results.setdefault("by_task", {}).setdefault(item["task"], {"c": 0, "t": 0})
        results["by_task"][item["task"]]["t"] += 1
        results["by_task"][item["task"]]["c"] += int(correct)

    results["generalization_score"] = results["correct"] / results["total"]

    print(f"\n{'=' * 50}")
    print(f"Generalization Score [{label}]")
    print(f"{'=' * 50}")
    print(f"Holdout file: {HOLDOUT_PATH}")
    print(f"Overall: {results['correct']}/{results['total']} = {results['generalization_score']:.1%}")
    for tsk, r in results.get("by_task", {}).items():
        print(f"  {tsk}: {r['c']}/{r['t']} = {r['c'] / r['t']:.1%}")

    return results


if __name__ == "__main__":
    from agentguard_gym.grandfinals.adversarial_loop import DefenderBackend

    print("Running baseline evaluation on data/holdout_attacks.json ...")
    backend = DefenderBackend()
    evaluate_on_held_out(lambda p: backend.decide(observation_text=p), "BEFORE TRAINING (Heuristic Baseline)")
