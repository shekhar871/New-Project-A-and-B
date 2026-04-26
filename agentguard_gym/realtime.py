from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple

from agentguard_gym.environment import AgentGuardEnvironment
from agentguard_gym.models import CyberTaskType, DefenseActionType


EventType = Literal[
    "trainer.started",
    "trainer.stopped",
    "trainer.tick",
    "trainer.metrics",
    "episode.started",
    "episode.step",
    "episode.ended",
    "error",
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


@dataclass
class TrainerConfig:
    policy: Literal["random", "heuristic"] = "heuristic"
    steps_per_episode: int = 12
    episodes_per_minute: int = 30
    seed: int = 42


@dataclass
class TrainerStatus:
    running: bool = False
    policy: str = "heuristic"
    started_at_ms: Optional[int] = None
    episodes: int = 0
    steps: int = 0
    mean_reward_100: float = 0.0
    last_episode_score: float = 0.0
    last_task: Optional[str] = None
    last_episode_id: Optional[str] = None


class EventBus:
    """
    Tiny in-process pub/sub for a real-time UI.

    We keep it intentionally dependency-free (no Redis/WebSockets required) and stream events via SSE.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10_000)

    def emit(self, typ: EventType, payload: Dict[str, Any]) -> None:
        evt = {"type": typ, "ts_ms": _now_ms(), **payload}
        try:
            self._q.put_nowait(evt)
        except queue.Full:
            # Drop oldest by draining a bit; keep UI responsive under bursty logs.
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(evt)
            except queue.Full:
                pass

    def stream(self) -> Generator[str, None, None]:
        """
        Server-Sent Events stream payload strings.
        """
        while True:
            evt = self._q.get()
            yield f"event: {evt['type']}\n"
            yield f"data: {_safe_json(evt)}\n\n"


def _random_policy_action(rng: Any) -> Dict[str, Any]:
    defense = rng.choice(list(DefenseActionType)).value
    return {"defense": defense, "rationale": f"random:{defense}"}


def _heuristic_policy_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic baseline to make the UI feel "smart" without any ML.

    - Prompt injection: block if obvious keyword, else allow.
    - SSRF: block if metadata IP, else audit tool chain (safe-ish).
    - Memory: quarantine/clear based on flags, else allow.
    """
    task = observation.get("task")
    artifacts = observation.get("artifacts") or []
    tool_trace = observation.get("tool_trace") or []

    if task == CyberTaskType.PROMPT_INJECTION.value:
        text = ""
        for a in artifacts:
            if a.get("type") == "prompt":
                text = str(a.get("content") or "")
        lower = text.lower()
        if "ignore" in lower or "exfiltrate" in lower or "/etc/passwd" in lower:
            return {"defense": DefenseActionType.BLOCK.value, "rationale": "heuristic:prompt_injection_keywords"}
        return {"defense": DefenseActionType.ALLOW.value, "rationale": "heuristic:benign_prompt"}

    if task == CyberTaskType.TOOL_MISUSE_SSRF.value:
        url = ""
        for a in artifacts:
            if a.get("type") == "url":
                url = str(a.get("url") or "")
        if "169.254.169.254" in url:
            return {"defense": DefenseActionType.BLOCK.value, "rationale": "heuristic:metadata_ip"}
        # auditing legitimate URLs is an ok “safe default” and yields partial credit on real threats
        return {"defense": DefenseActionType.AUDIT_TOOL_CHAIN.value, "rationale": "heuristic:audit_url"}

    # memory task (enum value is "memory_poisoning_privilege")
    flags: Dict[str, Any] = {}
    for a in artifacts:
        if a.get("type") == "memory_slot":
            flags = a.get("flags") or {}
            break
    if flags.get("env_secret_exposure_hint"):
        return {
            "defense": DefenseActionType.CLEAR_EXPOSED_SECRETS.value,
            "rationale": "heuristic:clear_exposed_secrets",
        }
    if flags.get("poisoned_vector_hint"):
        return {"defense": DefenseActionType.QUARANTINE_MEMORY.value, "rationale": "heuristic:quarantine_memory"}
    if any("process.env.get" in str(x) for x in tool_trace):
        return {"defense": DefenseActionType.AUDIT_TOOL_CHAIN.value, "rationale": "heuristic:audit_suspicious_trace"}
    return {"defense": DefenseActionType.ALLOW.value, "rationale": "heuristic:benign_memory"}


class RealTimeTrainer:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status = TrainerStatus()
        self._last_100: List[float] = []

    def status(self) -> TrainerStatus:
        with self._lock:
            return TrainerStatus(**self._status.__dict__)

    def start(self, cfg: TrainerConfig) -> TrainerStatus:
        with self._lock:
            if self._status.running:
                return TrainerStatus(**self._status.__dict__)
            self._stop.clear()
            self._status.running = True
            self._status.policy = cfg.policy
            self._status.started_at_ms = _now_ms()
            self._thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
            self._thread.start()
            snapshot = TrainerStatus(**self._status.__dict__)
        self._bus.emit("trainer.started", {"policy": cfg.policy, "policy_type": cfg.policy})
        return snapshot

    def stop(self) -> TrainerStatus:
        self._stop.set()
        with self._lock:
            self._status.running = False
        self._bus.emit("trainer.stopped", {})
        return self.status()

    def run_single_episode(
        self,
        *,
        task: Optional[CyberTaskType] = None,
        seed: Optional[int] = None,
        policy: Literal["random", "heuristic"] = "heuristic",
        steps_limit: int = 12,
    ) -> Dict[str, Any]:
        env = AgentGuardEnvironment()
        ep_id = str(uuid.uuid4())
        obs = env.reset(seed=seed, episode_id=ep_id, task=task)
        trace: List[Dict[str, Any]] = []
        rewards: List[float] = []

        import random as _random

        rng = _random.Random(seed if seed is not None else 0)

        for _ in range(steps_limit):
            obs_dict = obs.model_dump(mode="json")
            if policy == "random":
                action = _random_policy_action(rng)
            else:
                action = _heuristic_policy_action(obs_dict)
            res = env.step(action)
            rewards.append(float(res.reward.value))
            trace.append(
                {
                    "observation": obs_dict,
                    "action": action,
                    "reward": res.reward.model_dump(mode="json"),
                    "done": res.done,
                    "info": res.info,
                }
            )
            obs = res.observation
            if res.done:
                break

        score = sum(rewards) / max(1, len(rewards))
        return {"episode_id": ep_id, "task": (task.value if task else None), "policy": policy, "score": score, "trace": trace}

    def _run(self, cfg: TrainerConfig) -> None:
        import random as _random

        env = AgentGuardEnvironment()
        rng = _random.Random(cfg.seed)
        sleep_s = max(0.01, 60.0 / max(1, cfg.episodes_per_minute))

        while not self._stop.is_set():
            try:
                task = rng.choice(list(CyberTaskType))
                ep_id = str(uuid.uuid4())
                obs = env.reset(seed=cfg.seed, episode_id=ep_id, task=task)
                self._bus.emit("episode.started", {"episode_id": ep_id, "task": task.value})

                rewards: List[float] = []
                for step in range(cfg.steps_per_episode):
                    obs_dict = obs.model_dump(mode="json")
                    if cfg.policy == "random":
                        action = _random_policy_action(rng)
                    else:
                        action = _heuristic_policy_action(obs_dict)
                    res = env.step(action)
                    r = float(res.reward.value)
                    rewards.append(r)
                    self._bus.emit(
                        "episode.step",
                        {
                            "episode_id": ep_id,
                            "task": task.value,
                            "step": step + 1,
                            "action": action,
                            "reward": r,
                            "outcome": res.reward.outcome,
                            "partial_credit": res.reward.partial_credit,
                            "done": res.done,
                            "policy_type": cfg.policy,
                        },
                    )
                    obs = res.observation
                    with self._lock:
                        self._status.steps += 1
                    if res.done:
                        break

                score = sum(rewards) / max(1, len(rewards))
                self._last_100.append(score)
                if len(self._last_100) > 100:
                    self._last_100 = self._last_100[-100:]
                mean_100 = sum(self._last_100) / max(1, len(self._last_100))

                with self._lock:
                    self._status.episodes += 1
                    self._status.last_episode_score = score
                    self._status.mean_reward_100 = mean_100
                    self._status.last_task = task.value
                    self._status.last_episode_id = ep_id

                self._bus.emit(
                    "episode.ended",
                    {"episode_id": ep_id, "task": task.value, "score": score, "mean_score_100": mean_100},
                )
                self._bus.emit("trainer.tick", {"episodes": self._status.episodes, "mean_score_100": mean_100})
                time.sleep(sleep_s)
            except Exception as exc:  # noqa: BLE001
                self._bus.emit("error", {"message": str(exc)})
                time.sleep(0.25)

