#!/usr/bin/env python3
"""
G14 — demo cache generator.

Writes `results/demo_cache.json` with a fixed set of adversarial episodes so the live demo
does not depend on API keys or runtime generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentguard_gym.grandfinals.adversarial_loop import AdversarialSystem
from agentguard_gym.models import CyberAdversarialTaskType


def main() -> int:
    data_dir = Path("data")
    seed_path = data_dir / "attack_corpus.json"
    if not seed_path.is_file():
        raise SystemExit("Missing data/attack_corpus.json")
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))

    sys = AdversarialSystem.from_seed_attacks(seeds, data_dir=data_dir)
    cache = []
    for task in (
        CyberAdversarialTaskType.PROMPT_INJECTION,
        CyberAdversarialTaskType.TOOL_MISUSE_SSRF,
        CyberAdversarialTaskType.MEMORY_POISONING,
    ):
        for i in range(10):
            ep = sys.run_episode(task=task, step_idx=i)
            cache.append(ep.model_dump(mode="json"))

    out = Path("results/demo_cache.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": cache}, indent=2), encoding="utf-8")
    print(f"Cached {len(cache)} episodes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
