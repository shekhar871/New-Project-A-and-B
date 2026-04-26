from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReplayItem:
    prompt: str
    completion: str
    reward: float
    outcome: str
    task: str
    difficulty: float = 0.0
    priority: float = 1.0


class ExperienceBuffer:
    """
    Training-time experience replay buffer.

    Stores (prompt, completion, reward, outcome, task, difficulty) samples.
    Supports uniform sampling and a simple prioritized replay heuristic.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self._buf: Deque[ReplayItem] = deque(maxlen=max(1, int(capacity)))
        self._lock = threading.Lock()

    def add(self, item: ReplayItem) -> None:
        with self._lock:
            self._buf.append(item)

    def __len__(self) -> int:  # noqa: D105
        with self._lock:
            return len(self._buf)

    def mean_reward(self) -> float:
        with self._lock:
            if not self._buf:
                return 0.0
            return sum(x.reward for x in self._buf) / len(self._buf)

    def sample_uniform(self, n: int) -> List[ReplayItem]:
        with self._lock:
            pool = list(self._buf)
        if not pool or n <= 0:
            return []
        k = min(int(n), len(pool))
        return random.sample(pool, k)

    def sample_prioritized(self, n: int) -> List[ReplayItem]:
        """
        Lightweight PER-style replay:
        priority is stored on insert; higher priority more likely.
        """
        with self._lock:
            pool = list(self._buf)
        if not pool or n <= 0:
            return []
        weights = [max(1e-6, float(x.priority)) for x in pool]
        k = min(int(n), len(pool))
        return random.choices(pool, weights=weights, k=k)


EXPERIENCE_BUFFER = ExperienceBuffer()


class ReplayMixedPrompts:
    """
    A small wrapper that exposes a "dataset-like" interface for TRL:
    - base_prompts: fixed offline prompts (records)
    - replay: ExperienceBuffer prompts sampled stochastically

    We intentionally ignore the provided index and sample with replacement
    to break temporal correlation and inject replay diversity.
    """

    def __init__(
        self,
        *,
        base_prompts: Sequence[Dict[str, str]],
        replay_ratio: float = 0.30,
        prioritized: bool = True,
    ) -> None:
        self._base = list(base_prompts)
        self._replay_ratio = float(max(0.0, min(1.0, replay_ratio)))
        self._prioritized = bool(prioritized)

    def __len__(self) -> int:  # noqa: D105
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict[str, str]:  # noqa: D105
        # If replay has data, sometimes return replayed prompt instead of fresh.
        if len(EXPERIENCE_BUFFER) > 0 and random.random() < self._replay_ratio:
            items = (
                EXPERIENCE_BUFFER.sample_prioritized(1)
                if self._prioritized
                else EXPERIENCE_BUFFER.sample_uniform(1)
            )
            if items:
                return {"prompt": items[0].prompt}
        # "fresh" path: sample from offline corpus (not sequential)
        return random.choice(self._base)

