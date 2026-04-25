from __future__ import annotations

"""
Grand Finals requirement: an end-to-end training pipeline (not static-only).

This script pre-generates a mixed corpus of malicious and benign scenarios into JSONL:
  data/offline_corpus.jsonl

Why this exists:
- avoids live API rate-limit failures mid-training (gap G9)
- makes training reproducible and debuggable

If `ANTHROPIC_API_KEY` is set and `anthropic` is installed, it will use Anthropic.
Otherwise it will use the seed corpus with deterministic obfuscations.
"""

import json
import random
from pathlib import Path

from agentguard_gym.grandfinals.adversarial_loop import AdversarialSystem
from agentguard_gym.models import CyberAdversarialTaskType


def main() -> None:
    seed_path = Path("data/attack_corpus.json")
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    system = AdversarialSystem.from_seed_attacks(seeds)

    benign_path = Path("data/benign_corpus.json")
    if not benign_path.is_file():
        raise SystemExit("Missing data/benign_corpus.json — add it to mix TN/FP in GRPO training.")
    benign_data = json.loads(benign_path.read_text(encoding="utf-8"))

    out_path = Path("data/offline_corpus.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 600 scenarios total target (as per your gap file): 200 per task
    per_task = 200
    benign_ratio = 0.30  # Aligned with live AdversarialSystem (BENIGN_EPISODE_PROB)

    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for task in (
            CyberAdversarialTaskType.PROMPT_INJECTION,
            CyberAdversarialTaskType.TOOL_MISUSE_SSRF,
            CyberAdversarialTaskType.MEMORY_POISONING,
        ):
            for i in range(per_task):
                is_benign = (i / per_task) < benign_ratio
                if is_benign:
                    brows = benign_data.get(task.value) or []
                    if not brows:
                        raise SystemExit(f"benign_corpus.json missing entries for {task.value}")
                    row = random.choice(brows)
                    record = {
                        "task": task.value,
                        "attack_text": row.get("text", ""),
                        "attack_type": "benign",
                        "difficulty_score": 0.0,
                        "is_malicious_ground_truth": False,
                    }
                else:
                    a = system.attacker.generate(task=task, difficulty=min(1.0, 0.1 + (i / per_task)))
                    record = {
                        "task": task.value,
                        "attack_text": a.attack_text,
                        "attack_type": a.attack_type,
                        "difficulty_score": a.difficulty_score,
                        "is_malicious_ground_truth": True,
                    }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1

    print(f"Wrote {rows} records to {out_path}")


if __name__ == "__main__":
    main()

