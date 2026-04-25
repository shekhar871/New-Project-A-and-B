from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Optional: Sentence-BERT (blueprint) — falls back to bag-of-words
_EMBEDDER = None
_USE_EMBED: Optional[bool] = None


def _get_embedder():
    global _EMBEDDER, _USE_EMBED
    if _USE_EMBED is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        _USE_EMBED = True
    except Exception:
        _EMBEDDER = None
        _USE_EMBED = False
    return _EMBEDDER


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


def _bow_novelty(text: str, corpus_texts: Iterable[str]) -> Tuple[float, float]:
    a = _bow(text)
    best = 0.0
    for c in corpus_texts:
        best = max(best, cosine_sim_bow(a, _bow(c)))
    return max(0.0, min(1.0, 1.0 - best)), best


def _embed_novelty(text: str, corpus_list: list[str]) -> Tuple[float, float]:
    import numpy as np

    m = _get_embedder()
    if m is None or not corpus_list:
        return _bow_novelty(text, corpus_list)
    embs = m.encode(
        corpus_list,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    q = m.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    sims = np.dot(embs, q)
    best = float(np.max(sims)) if sims.size else 0.0
    return max(0.0, min(1.0, 1.0 - best)), best


def novelty_score(text: str, corpus_texts: Iterable[str]) -> Tuple[float, float]:
    """
    Returns (novelty, max_similarity).

    Uses SentenceTransformers **all-MiniLM-L6-v2** cosine when `sentence-transformers` is
    installed; otherwise bag-of-words (dependency-free, portable).
    """
    _get_embedder()
    corpus_list = [c for c in corpus_texts if c and str(c).strip()]
    if _USE_EMBED and corpus_list:
        try:
            return _embed_novelty(text, corpus_list)
        except Exception:
            pass
    return _bow_novelty(text, corpus_list or [])
