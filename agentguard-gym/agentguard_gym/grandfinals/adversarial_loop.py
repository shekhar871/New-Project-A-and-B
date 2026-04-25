from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentguard_gym.models import (
    AdversarialEpisodeResult,
    AttackerAction,
    CyberAdversarialTaskType,
    ELOTracker,
)
from agentguard_gym.grandfinals.attacker import Attacker, AttackCorpus
from agentguard_gym.grandfinals.curriculum import CurriculumManager
from agentguard_gym.grandfinals.judge import Judge
from agentguard_gym.grandfinals.novelty import novelty_score
from agentguard_gym.grandfinals.observation_builder import build_defender_observation_narrative
from agentguard_gym.grandfinals.prompts import DEFENDER_SYSTEM_PROMPT
from agentguard_gym.grandfinals.rewards import (
    attacker_reward,
    defender_reward_normalize,
    defender_reward_raw,
    safe_reward,
)


def _parse_defender_decision(completion: str) -> Optional[str]:
    match = re.search(r'"decision"\s*:\s*"(\w+)"', completion)
    if match:
        v = match.group(1).lower()
        if v in ("block", "allow"):
            return v
    return None


def _json_object_from_text(text: str) -> Optional[dict]:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group())
    except Exception:
        return None


BENIGN_EPISODE_PROB = 0.30


def _package_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


def _load_benign_corpus(data_dir: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    if data_dir is not None:
        candidates = [data_dir / "benign_corpus.json"]
    else:
        candidates = [Path("data") / "benign_corpus.json", _package_data_dir() / "benign_corpus.json"]
    p: Optional[Path] = next((c for c in candidates if c.is_file()), None)
    if p is None:
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): v for k, v in raw.items() if isinstance(v, list)}


class DefenderBackend:
    """
    Defender policy backend for episodes:
    - if HF/OpenAI-compatible vars are present, uses OpenAI client to get JSON decision.
    - else uses deterministic heuristic (same as UI trainer, but blueprint schema).
    """

    def __init__(self) -> None:
        import os

        self.api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.api_base_url = os.getenv("API_BASE_URL")
        self.model_name = os.getenv("MODEL_NAME")

    def decide(self, *, observation_text: str) -> str:
        if self.api_key and self.api_base_url and self.model_name:
            return self._llm_decide(observation_text=observation_text)
        return self._heuristic_decide(observation_text=observation_text)

    def _llm_decide(self, *, observation_text: str) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
                {"role": "user", "content": observation_text},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _heuristic_decide(self, *, observation_text: str) -> str:
        lower = observation_text.lower()
        # Legitimate traffic is labeled [GT:benign] — do not "always block" on weak keywords
        if re.search(r"\[gt:benign\]", lower):
            if (
                "169.254.169.254" in lower
                or "metadata.google.internal" in lower
                or re.search(r"ignore (all|previous|instructions)", lower)
                or "system override" in lower
                or "jailbreak" in lower
                or "exfiltrate" in lower
            ):
                return json.dumps(
                    {
                        "attacker_intent_inference": "Benign-tagged but contains strong internal or override patterns.",
                        "decision": "block",
                        "confidence": 0.8,
                        "threat_type": "heuristic_mixed_signal",
                        "reasoning": "Override or cloud-metadata patterns on a benign-tagged event — block pending human review.",
                    }
                )
            return json.dumps(
                {
                    "attacker_intent_inference": "Normal user or operational traffic.",
                    "decision": "allow",
                    "confidence": 0.7,
                    "threat_type": None,
                    "reasoning": "Ground truth tagged benign; no high-confidence malicious indicators.",
                }
            )
        # Malicious (or unlabeled) path — naive high-risk patterns
        if "169.254.169.254" in lower or "metadata" in lower or "ignore" in lower or "system" in lower:
            return json.dumps(
                {
                    "attacker_intent_inference": "Likely malicious attempt to override behavior or access restricted resources.",
                    "decision": "block",
                    "confidence": 0.85,
                    "threat_type": "heuristic",
                    "reasoning": "Detected high-risk patterns consistent with known agentic attacks.",
                }
            )
        return json.dumps(
            {
                "attacker_intent_inference": "No malicious intent detected.",
                "decision": "allow",
                "confidence": 0.55,
                "threat_type": None,
                "reasoning": "No obvious high-risk markers found; allow by default.",
            }
        )


@dataclass
class AdversarialSystem:
    seed_attacks: List[dict]
    corpus: AttackCorpus
    attacker: Attacker
    judge: Judge
    curriculum: CurriculumManager
    elo: ELOTracker
    defender: DefenderBackend
    benign_by_task: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_seed_attacks(
        cls,
        seed_attacks: List[dict],
        *,
        benign_corpus: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        data_dir: Optional[Path] = None,
    ) -> "AdversarialSystem":
        corpus = AttackCorpus(entries=[])
        attacker = Attacker(corpus=corpus, seed_attacks=seed_attacks)
        ben: Dict[str, List[Dict[str, Any]]] = {}
        if benign_corpus is not None:
            ben = benign_corpus
        else:
            ben = _load_benign_corpus(data_dir)
        return cls(
            seed_attacks=seed_attacks,
            corpus=corpus,
            attacker=attacker,
            judge=Judge(),
            curriculum=CurriculumManager(),
            elo=ELOTracker(),
            defender=DefenderBackend(),
            benign_by_task=ben,
        )

    def run_episode(
        self,
        *,
        task: Optional[CyberAdversarialTaskType] = None,
        step_idx: int = 0,
        episode_id: Optional[str] = None,
        generalization_attack: Optional[AttackerAction] = None,
    ) -> AdversarialEpisodeResult:
        ep_id = episode_id or str(uuid.uuid4())
        if task is None:
            task = random.choice(list(CyberAdversarialTaskType))

        difficulty = 0.10 if self.curriculum.is_warmup() else float(self.curriculum.difficulty)
        bkey = task.value
        benign_list = self.benign_by_task.get(bkey) or []
        use_benign = (
            generalization_attack is None
            and bool(benign_list)
            and (random.random() < BENIGN_EPISODE_PROB)
        )
        if use_benign:
            return self._run_benign_episode(
                task=task,
                step_idx=step_idx,
                episode_id=ep_id,
                difficulty=difficulty,
                benign_list=benign_list,
            )

        if generalization_attack is not None:
            attack = generalization_attack
            is_gen = True
            novelty = 1.0
            max_sim = 0.0
        else:
            attack = self.attacker.generate(task=task, difficulty=difficulty)
            novelty, max_sim = novelty_score(attack.attack_text, self.corpus.texts()[:-1])
            is_gen = False

        obs = build_defender_observation_narrative(
            attack=attack,
            task=task,
            is_malicious=True,
            step_idx=step_idx,
            episode_id=ep_id,
        )

        completion = self.defender.decide(observation_text=obs)
        decision = _parse_defender_decision(completion) or "allow"

        blocked = decision == "block"
        outcome = "TP" if blocked else "FN"
        defender_won = outcome in ("TP", "TN")

        r_format = 1.0 if decision in ("block", "allow") else 0.0
        raw_d = defender_reward_raw(outcome=outcome, task=task.value, step_idx=step_idx, r_format=r_format)
        r_d = defender_reward_normalize(raw_d)

        verdict = self.judge.evaluate(attack=attack, defender_decision=decision, outcome=outcome)

        rep_penalty = 1.0 if max_sim > 0.85 else 0.0
        r_a = attacker_reward(
            novelty=float(novelty),
            difficulty=float(self.elo.difficulty()),
            defender_success=1.0 if defender_won else 0.0,
            judge_quality=float(verdict.attack_sophistication),
            repetition_penalty=rep_penalty,
        )

        self.elo.update(defender_won=defender_won)
        self.curriculum.update(defender_won=defender_won, reward_d=r_d, reward_a=r_a)

        return AdversarialEpisodeResult(
            episode_id=ep_id,
            task=task,
            difficulty_level=float(difficulty),
            attack=attack,
            defender_decision=decision,  # type: ignore[arg-type]
            outcome=outcome,  # type: ignore[arg-type]
            defender_won=bool(defender_won),
            is_malicious_scenario=True,
            reward_defender=float(r_d),
            reward_attacker=float(r_a),
            novelty_score=float(novelty),
            judge_verdict=verdict,
            generalization_test=bool(is_gen),
        )

    def _run_benign_episode(
        self,
        *,
        task: CyberAdversarialTaskType,
        step_idx: int,
        episode_id: str,
        difficulty: float,
        benign_list: List[Dict[str, Any]],
    ) -> AdversarialEpisodeResult:
        payload = random.choice(benign_list)
        text = str(payload.get("text", "")).strip()
        if len(text) < 5:
            text = "placeholder benign content"
        attack = AttackerAction(
            attack_text=text,
            attack_type="benign_legitimate",
            difficulty_score=0.05,
            target_task=task,
            reasoning="benign_corpus",
            low_novelty=True,
        )
        obs = build_defender_observation_narrative(
            attack=attack,
            task=task,
            is_malicious=False,
            step_idx=step_idx,
            episode_id=episode_id,
        )
        completion = self.defender.decide(observation_text=obs)
        decision = _parse_defender_decision(completion) or "allow"
        blocked = decision == "block"
        outcome: str = "FP" if blocked else "TN"
        defender_won = outcome in ("TP", "TN")

        r_format = 1.0 if decision in ("block", "allow") else 0.0
        raw_d = defender_reward_raw(outcome=outcome, task=task.value, step_idx=step_idx, r_format=r_format)
        r_d = defender_reward_normalize(raw_d)

        verdict = self.judge.evaluate(attack=attack, defender_decision=decision, outcome=outcome)

        r_a = attacker_reward(
            novelty=0.0,
            difficulty=float(self.elo.difficulty()),
            defender_success=1.0 if defender_won else 0.0,
            judge_quality=float(verdict.attack_sophistication),
            repetition_penalty=0.0,
        )

        self.elo.update(defender_won=defender_won)
        self.curriculum.update(defender_won=defender_won, reward_d=r_d, reward_a=r_a)

        return AdversarialEpisodeResult(
            episode_id=episode_id,
            task=task,
            difficulty_level=float(difficulty),
            attack=attack,
            defender_decision=decision,  # type: ignore[arg-type]
            outcome=outcome,  # type: ignore[arg-type]
            defender_won=bool(defender_won),
            is_malicious_scenario=False,
            reward_defender=float(r_d),
            reward_attacker=float(r_a),
            novelty_score=0.0,
            judge_verdict=verdict,
            generalization_test=False,
        )

