# agentguard_gym/training/callbacks.py
# Based on: REPO (2025), GTPO (2025), CE-GPPO (2025), Entropy Scheduling paper (2025)

import json
import math
from pathlib import Path
from typing import Any, Optional

from transformers import TrainerCallback, TrainerState, TrainerControl

from agentguard_gym.training.reward_fns import get_outcome_metrics


class JsonlTrainingMetricsCallback(TrainerCallback):
    """
    Append one JSON line per on_log to runs/training_metrics.jsonl for offline plots
    (scripts/plot_training_metrics.py) when judges cannot access W&B.
    """

    def __init__(self, path: str = "runs/training_metrics.jsonl") -> None:
        self.path = path

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if not logs:
            return control
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        row: dict = {
            "step": int(state.global_step),
            "entropy": logs.get("entropy") or logs.get("train/entropy"),
            "loss": logs.get("loss") or logs.get("train/loss"),
        }
        row.update({k: float(v) for k, v in get_outcome_metrics().items() if k in ("fp_rate", "win_rate")})
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return control


class WandbOutcomeMetricsCallback(TrainerCallback):
    """
    Logs confusion rates from GRPO reward_fn tallies: fp_rate, win_rate, outcome/*.
    """

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        try:
            import wandb
        except Exception:
            return control
        if wandb.run is None:
            return control
        m = get_outcome_metrics()
        if not m:
            return control
        wandb.log({**m, "train/global_step": float(state.global_step)})
        return control

class EntropyGuardCallback(TrainerCallback):
    """
    3-layer entropy protection:
    1. KL annealing (REPO / Entropy Scheduling, 2025)
    2. Collapse detection + emergency stop (GTPO, 2025)
    3. Logging for demo curve
    """
    def __init__(self, beta_0=0.04, decay=0.5, t_max=200, h_min_ratio=0.5):
        self.beta_0 = beta_0
        self.decay = decay
        self.t_max = t_max
        self.h_min_ratio = h_min_ratio
        self.h_initial = None
        self.history = []
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        t = state.global_step
        
        # Layer 1: KL annealing
        new_beta = max(0.015, self.beta_0 * math.exp(-self.decay * t / self.t_max))
        if hasattr(args, 'beta'):
            args.beta = new_beta
        
        if t % 10 == 0:
            print(f"[Step {t}] β={new_beta:.4f} | ELO_A={kwargs.get('elo_a','?')} | ELO_D={kwargs.get('elo_d','?')}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        
        entropy = logs.get("entropy") or logs.get("train/entropy")
        if entropy is None:
            return control
        
        self.history.append({"step": state.global_step, "entropy": entropy})
        
        if self.h_initial is None:
            self.h_initial = entropy
            return control
        
        # Layer 2: Collapse detection (GTPO criterion)
        if entropy < self.h_min_ratio * self.h_initial:
            print(f"⚠️  ENTROPY COLLAPSE at step {state.global_step}!")
            print(f"   entropy={entropy:.3f} < {self.h_min_ratio}×{self.h_initial:.3f}")
            control.should_save = True
            # Emergency: reduce LR
            for pg in kwargs.get("optimizer", {}).param_groups if kwargs.get("optimizer") else []:
                pg['lr'] *= 0.5
                print(f"   LR reduced to {pg['lr']:.2e}")
        
        return control
