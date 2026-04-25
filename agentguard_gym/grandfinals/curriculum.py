from __future__ import annotations

from collections import deque
from statistics import mean, stdev
from typing import Deque, List, Tuple


class CurriculumManager:
    """
    Grand Finals blueprint curriculum + Nash detector.

    - Maintains difficulty in [0,1]
    - Tracks rolling win rate
    - Detects equilibrium (low variance) and forces a phase shift
    """

    def __init__(self) -> None:
        self.difficulty: float = 0.20
        self._warmup_remaining: int = 10
        self.win_history: Deque[float] = deque(maxlen=10)
        self.reward_history: Deque[Tuple[float, float]] = deque(maxlen=20)
        self.phase_transitions: List[int] = []
        self.episodes_seen: int = 0

    def is_warmup(self) -> bool:
        return self._warmup_remaining > 0

    def update(self, *, defender_won: bool, reward_d: float, reward_a: float) -> bool:
        """
        Returns True when a Nash-like equilibrium is detected and a phase transition was triggered.
        """
        self.episodes_seen += 1
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return False

        self.win_history.append(1.0 if defender_won else 0.0)
        self.reward_history.append((float(reward_d), float(reward_a)))

        # Win-rate based difficulty adjustments
        if len(self.win_history) >= 10:
            win = mean(self.win_history)
            if win > 0.85:
                self.difficulty = min(1.0, self.difficulty + 0.10)
            elif win < 0.40:
                self.difficulty = max(0.10, self.difficulty - 0.10)

        # Nash detector (variance collapse)
        if len(self.reward_history) == 20:
            sigma_d = stdev([x for x, _ in self.reward_history])
            sigma_a = stdev([y for _, y in self.reward_history])
            if sigma_d < 0.05 and sigma_a < 0.05:
                self.difficulty = min(1.0, self.difficulty + 0.20)
                self.phase_transitions.append(self.episodes_seen)
                self.reward_history.clear()
                return True

        return False

    def status(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "win_rate_10": (mean(self.win_history) if self.win_history else None),
            "episodes_seen": self.episodes_seen,
            "phase_transitions": self.phase_transitions,
            "warmup_remaining": self._warmup_remaining,
        }

