# agentguard_gym/training/callbacks.py
# Based on: REPO (2025), GTPO (2025), CE-GPPO (2025), Entropy Scheduling paper (2025)

import math
from transformers import TrainerCallback, TrainerState, TrainerControl

from agentguard_gym.training.metrics_logger import JsonlMetricsLogger

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
        self._logger = JsonlMetricsLogger()
        self._last_beta = float(beta_0)
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        t = state.global_step
        
        # Layer 1: KL annealing
        new_beta = max(0.015, self.beta_0 * math.exp(-self.decay * t / self.t_max))
        if hasattr(args, 'beta'):
            args.beta = new_beta
        self._last_beta = float(new_beta)
        
        if t % 10 == 0:
            print(f"[Step {t}] β={new_beta:.4f} | ELO_A={kwargs.get('elo_a','?')} | ELO_D={kwargs.get('elo_d','?')}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        
        entropy = logs.get("entropy") or logs.get("train/entropy")
        if entropy is None:
            return control
        
        self.history.append({"step": state.global_step, "entropy": entropy})
        try:
            self._logger.log(
                {
                    "step": int(state.global_step),
                    "entropy": float(entropy),
                    "beta": float(self._last_beta),
                    "mean_reward": float(logs.get("mean_reward", logs.get("reward", 0.0)) or 0.0),
                }
            )
        except Exception:
            pass
        
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
