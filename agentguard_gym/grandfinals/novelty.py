from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9:/._\\-\\s]+", " ", text)
    toks = [t for t in text.split() if t]
    return toks[:512]


def _bow(text: str) -> dict:
    d: dict = {}
    for t in _tokenize(text):
        d[t] = d.get(t, 0) + 1
    return d


def cosine_sim_bow(a: dict, b: dict) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, va in a.items():
        na += float(va * va)
        vb = b.get(k)
        if vb is not None:
            dot += float(va * vb)
    for vb in b.values():
        nb += float(vb * vb)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def novelty_score(text: str, corpus_texts: Iterable[str]) -> Tuple[float, float]:
    """
    Returns (novelty, max_similarity).

    Novelty = 1 - max_cosine_similarity(bag-of-words).
    This is dependency-free and deterministic; you can swap in SentenceTransformers later.
    """
    a = _bow(text)
    best = 0.0
    for c in corpus_texts:
        best = max(best, cosine_sim_bow(a, _bow(c)))
    return max(0.0, min(1.0, 1.0 - best)), best

