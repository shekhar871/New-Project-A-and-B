# scripts/run_demo.py
# Simulates episodes to trigger Phase Shift

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agentguard_gym.grandfinals.adversarial_loop import AdversarialSystem
from agentguard_gym.models import CyberAdversarialTaskType
import json

def run():
    print("Loading system...")
    seeds = json.loads(Path("data/attack_corpus.json").read_text(encoding="utf-8"))
    sys = AdversarialSystem.from_seed_attacks(seeds)
    
    print("Running 65 episodes to trigger Phase Transition...")
    for i in range(1, 66):
        res = sys.run_episode(step_idx=i)
        win_rate = sys.curriculum.status().get("win_rate_10")
        win_rate_str = f"{win_rate:.2f}" if win_rate is not None else "N/A"
        print(f"Step {i:02d} | Task: {res.task.value:<16} | Diff: {res.difficulty_level:.2f} | D_won: {res.defender_won} | WinRate: {win_rate_str}")
        
        # Manually force some reward patterns to trigger Nash check easily if we want 
        # But curriculum is automatic if using real models. Since we use heuristic fallback, 
        # the win rate might oscillate. The curriculum logs "PHASE TRANSITION" if std < 0.05.

if __name__ == "__main__":
    run()
