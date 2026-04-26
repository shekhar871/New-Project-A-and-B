from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Optional: Sentence-BERT (blueprint) — falls back to bag-of-words
_EMBEDDER = None
_USE_EMBED: Optional[bool] = None

# Optional embedding cache (R06)
_CORPUS_EMBS = None
_CORPUS_HASH: Optional[str] = None
_CACHE_DIR = Path("data")
_EMB_CACHE = _CACHE_DIR / "offline_embeddings.npy"
_HASH_CACHE = _CACHE_DIR / "offline_embeddings.sha256"

# Load cached embeddings immediately if present (fast path).
try:
    if _EMB_CACHE.exists() and _HASH_CACHE.exists():
        import numpy as np  # type: ignore

        _CORPUS_EMBS = np.load(_EMB_CACHE)
        _CORPUS_HASH = _HASH_CACHE.read_text(encoding="utf-8").strip() or None
except Exception:
    pass


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


def _corpus_sha(texts: List[str]) -> str:
    return hashlib.sha256("||".join(texts).encode("utf-8")).hexdigest()


def _load_or_encode_corpus(texts: List[str]):
    """
    Encode corpus ONCE and memoize; persist to .npy with a sha256 sidecar for invalidation.
    Returns normalized embeddings (np.float32).
    """
    import numpy as np

    global _CORPUS_EMBS, _CORPUS_HASH
    m = _get_embedder()
    if m is None or not texts:
        return None

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = _corpus_sha(texts)
    if _EMB_CACHE.exists() and _HASH_CACHE.exists():
        try:
            if _HASH_CACHE.read_text(encoding="utf-8").strip() == digest:
                _CORPUS_EMBS = np.load(_EMB_CACHE)
                _CORPUS_HASH = digest
                return _CORPUS_EMBS
        except Exception:
            pass

    embs = m.encode(
        texts,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    try:
        np.save(_EMB_CACHE, embs)
        _HASH_CACHE.write_text(digest, encoding="utf-8")
    except Exception:
        pass
    _CORPUS_EMBS = embs
    _CORPUS_HASH = digest
    return _CORPUS_EMBS


def _ensure_corpus_cache(corpus_list: List[str]):
    global _CORPUS_EMBS, _CORPUS_HASH
    if _CORPUS_EMBS is None:
        return _load_or_encode_corpus(corpus_list)
    digest = _corpus_sha(corpus_list)
    if _CORPUS_HASH != digest:
        return _load_or_encode_corpus(corpus_list)
    return _CORPUS_EMBS


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
    embs = _ensure_corpus_cache(corpus_list)
    if embs is None:
        return _bow_novelty(text, corpus_list)
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
